// The transform's tables and stage plan, the host side shared by the decoder's back half (transform.mjs) and the
// sender's encoder (../encoder.mjs, ../wgsl/inverse.mjs): the disc's layout of a picture's coefficients, the rows a
// version takes, the decode size, the detrend's basis and twiddles (transformTables), and the FFT's stages
// (stagesOf). Plain JS over the codec's wasm (sim/ob.mjs over liblizard/build/ob.mjs), so the sender's pages carry
// none of the decoder's kernels.
import { init, Lizard } from "../../sim/ob.mjs";
import { SIZES } from "../plan/common.mjs";
import { CHUNKS } from "../plan/back_tiers.mjs";

// Stages: the product is n; radix 8 wherever it divides what is left.
// ai: The first is radix 8 (firstStage); any n but 8 x 2^a 3^b 5^c is refused. The powers of two come first, then
// ai: each 5, then each 3, so 8 x 3 x 2^k keeps its one radix-3 stage last and its WGSL does not move. The last odd
// ai: stage runs in place (radix3Stage, oddLastStage), an earlier one through registers (oddStage): 1152 is 8 8 2 3 3,
// ai: 1280 is 8 8 4 5.
export function stagesOf(n) {
  const s = [];
  let m = n;
  for (; m > 1 && m % 2 === 0;) {
    const r = m % 8 === 0 ? 8 : m % 4 === 0 ? 4 : 2;
    if (s.length === 0 && r !== 8) break;
    s.push(r); m /= r;
  }
  for (; s.length && m % 5 === 0; m /= 5) s.push(5);
  for (; s.length && m % 3 === 0; m /= 3) s.push(3);
  if (m !== 1 || !s.length) throw new Error(`no transform of ${n} points: 8 x 2^a 3^b 5^c`);
  return s;
}

// The coefficient at pos p, as (u, v): the C's half-plane block layout (src/lizard_phy.c init).
export function uvOf(p, n, bw) {
  const v = Math.floor(p / (n * bw)) * bw + ((p % (n * bw)) % bw), uw = Math.floor((p % (n * bw)) / bw);
  return [uw < n / 2 ? uw : uw - n, v];
}

// S's rows (DESIGN section 3): row v holds u = ustart .. umaxP then -umaxN .. -1, ustart 1 on v = 0 and 0 elsewhere,
// packed at off[v]. rows[v] = (off, count, nonnegative entries, umaxN) for the shader; UV[i] is the S entry of pos[i]
// for the soft-value stage. Every entry of pos maps to exactly one entry of S, and S has npos entries.
// ai: UC[i] is pos[i]'s (u, v) itself, u in the high half (signed) and v in the low, for the soft stage's turn of a
// ai: coefficient by the grid's shift (glsl/back_tiers.mjs talign).
export function discLayout(shape, pos) {
  const { n, bw, npos } = shape;
  const byV = new Map();
  for (let i = 0; i < npos; i++) { const [u, v] = uvOf(pos[i], n, bw); if (!byV.has(v)) byV.set(v, []); byV.get(v).push(u); }
  const V = Math.max(...byV.keys()) + 1;
  const rows = new Uint32Array(4 * V), off = new Uint32Array(V), np = new Uint32Array(V), un = new Uint32Array(V);
  let count = 0;
  for (let v = 0; v < V; v++) {
    const us = byV.get(v) ?? [], set = new Set(us), ustart = v === 0 ? 1 : 0;
    const maxP = Math.max(ustart - 1, ...us.filter((u) => u >= 0)), maxN = Math.max(0, ...us.filter((u) => u < 0).map((u) => -u));
    let ok = us.length === maxP + 1 - ustart + maxN;
    for (let u = ustart; u <= maxP && ok; u++) ok = set.has(u);
    for (let u = -maxN; u < 0 && ok; u++) ok = set.has(u);
    if (!ok) throw new Error(`disc row v = ${v} of ${n}/${npos / 320} is not u = ${ustart}..${maxP}, -${maxN}..-1`);
    off[v] = count; np[v] = maxP + 1 - ustart; un[v] = maxN;
    rows.set([count, us.length, np[v], maxN], 4 * v);
    count += us.length;
  }
  if (count !== npos) throw new Error(`disc holds ${count} entries for ${npos} coefficients`);
  const UV = new Uint32Array(npos), UC = new Uint32Array(npos);
  for (let i = 0; i < npos; i++) {
    const [u, v] = uvOf(pos[i], n, bw), ustart = v === 0 ? 1 : 0;
    UV[i] = off[v] + (u >= 0 ? u - ustart : np[v] + un[v] + u);
    UC[i] = (((u & 0xffff) << 16) | (v & 0xffff)) >>> 0;
  }
  return { V, rows, off, UV, UC, count };
}

// ai: rowsBy[c], c = 0 .. blocks: the disc rows (1 + the largest v) that the coefficients of a frame's first c chunks
// ai: take, rowsBy[blocks] = V. Chunk b reads pos[i] for i in 8 b x 320 .. 8 (b + 1) x 320 (sub-channels 8b to
// ai: 8b + 7, 320 coefficients each, as glsl/back_tiers.mjs tsoft gathers them), and every version is a prefix of its
// ai: size's largest, so rowsOfVersion(tb, v), its first CHUNKS v chunks' rows, is version v's own V
// ai: (scripts/gpu/back/check_rows.mjs holds every version of every size to it).
export function rowsByCount({ n, bw, blocks, npos }, pos) {
  const per = npos / blocks, rowsBy = new Uint32Array(blocks + 1);
  let top = -1;
  for (let c = 1; c <= blocks; c++) {
    for (let i = per * (c - 1); i < per * c; i++) top = Math.max(top, uvOf(pos[i], n, bw)[1]);
    rowsBy[c] = top + 1;
  }
  return rowsBy;
}
// ai: The disc rows a frame of version v takes at a size's tables (transformTables): the gate's rows table by version.
export const rowsOfVersion = (tb, v) => tb.rowsBy[Math.min(CHUNKS * v, tb.blocks)];

// ai: The transform size a picture of n samples is read at by the fused pass 1: the ladder size below n, 256 below
// ai: 384 (2026-10-21). n is the first ladder size of at least 3 R, so the one below it holds 2 R and every coefficient
// ai: the picture carries; sampled on the picture's own square from its first sample, at n / nt of its step, a picture
// ai: band-limited to R gives the same coefficients at nt (each pass divides by its own size).
// ai: 1152 and 1280 (2026-10-29) at 1024, not at the ladder size below: their formats (LIZARD-576 to -880) were read at
// ai: 1024 from 1536 and keep that size and cost. 1024 holds 2R for both tops (LIZARD-720, R 383; -880, R 423), and
// ai: 1152 would be pass 1's third 64-lane wave (nt / 8 threads): measured on the iGPU, about 10% of the frame for 1.9
// ai: to 3.6% more blocks.
export const TRANSFORM_DOWN = { 384: 256, 512: 384, 768: 512, 1024: 768, 1152: 1024, 1280: 1024, 1536: 1024 };

// ai: The detrend's basis and its transforms at m points, as src/lizard_phy.c init defines them (in double, then f32):
// ai: cx, qx, then x12 (x1r, x1i, x2r, x2i a u), sx2 and sq2.
function detrendTables(m) {
  const cx = new Float32Array(m), qx = new Float32Array(m), x12 = new Float32Array(4 * m), er = new Float64Array(m), ei = new Float64Array(m);
  let m2 = 0, sq2 = 0;
  for (let i = 0; i < m; i++) { const c = (2 * i + 1 - m) / m; m2 += c * c; er[i] = Math.cos(-2 * 3.14159265358979323846 * i / m); ei[i] = Math.sin(-2 * 3.14159265358979323846 * i / m); }
  const sx2 = m2;
  m2 /= m;
  for (let i = 0; i < m; i++) { const c = (2 * i + 1 - m) / m, q = c * c - m2; cx[i] = c; qx[i] = q; sq2 += q * q; }
  for (let u = 0; u < m; u++) {
    let ar = 0, ai = 0, br = 0, bi = 0;
    for (let i = 0; i < m; i++) { const c = (2 * i + 1 - m) / m, q = c * c - m2, k = (u * i) % m; ar += c * er[k]; ai += c * ei[k]; br += q * er[k]; bi += q * ei[k]; }
    x12.set([ar, ai, br, bi], 4 * u);
  }
  return { cx, qx, x12, sx2: Math.fround(sx2), sq2: Math.fround(sq2) };
}

// ai: The tables for every picture size, from the codec: sizes = [{ n, subch, nt }] with a size's index its position
// ai: (gpu/tables.mjs PICTURES) and null where a size is not built. A picture's tables do not depend on the ring it
// ai: is painted in, so the codec is set up in its default ring. Besides A's own tables each entry carries what the
// ai: soft values bind to (UV, UC) and its blocks. Read once a build; the wasm holds one configuration at a time, so
// ai: this is not called while another Lizard is in use.
export async function transformTables(sizes) {
  await init();
  return sizes.map((z) => {
    if (!z) return null;
    const { n, subch } = z, nt = z.nt ?? n;
    if (SIZES.indexOf(n) < 0) throw new Error(`picture size ${n}`);
    const fc = new Lizard(n, subch, 1);
    // ai: blocks here are 8-sub-channel chunks, the rows' and the soft values' unit (a version is CHUNKS of them),
    // ai: which a profile does not change (2026-10-07: the format's rate profile, src/lizard_phy.h lizard_tiers_for, gives a
    // ai: frame fewer blocks than chunks, gpu/back/tiers.mjs)
    const shape = { ...fc.shape(), blocks: subch / 8 }, pos = fc.posTable(), tab = fc.tables();
    const disc = discLayout(shape, pos), rowsBy = rowsByCount(shape, pos);
    if (rowsBy[shape.blocks] !== disc.V) throw new Error(`${n}/${subch}: the blocks take ${rowsBy[shape.blocks]} rows of the disc's ${disc.V}`);
    // ai: the transform's tables at nt: the codec's own where nt is n, else the same definitions at nt
    let x12 = new Float32Array(4 * n), sx2 = tab[6 * n], sq2 = tab[6 * n + 1], cx = tab.slice(0, n), qx = tab.slice(n, 2 * n);
    for (let u = 0; u < n; u++) x12.set([tab[2 * n + u], tab[3 * n + u], tab[4 * n + u], tab[5 * n + u]], 4 * u);
    if (nt !== n) ({ x12, sx2, sq2, cx, qx } = detrendTables(nt));
    const tw = new Float32Array(2 * nt);
    for (let m = 0; m < nt; m++) { const th = -2 * Math.PI * m / nt; tw[2 * m] = Math.cos(th); tw[2 * m + 1] = Math.sin(th); }
    fc.free();
    return { n, subch, blocks: shape.blocks, shape, pos, V: disc.V, rowsBy, rows: disc.rows, off: disc.off, UV: disc.UV, UC: disc.UC, npos: shape.npos, x12, tw, sx2, sq2, m2: sx2 / nt, cx, qx, nt };
  });
}
