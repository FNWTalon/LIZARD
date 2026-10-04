// A: the picture to the spectrum's disc (gpu/back/DESIGN.md sections 1 to 3). Four shaders: GATE lists the batch's
// frames by picture size and writes the indirect args; PASS1 transforms picture columns along y, two real columns
// to a complex FFT, and leaves the column means and the detrend partials; REDUCE turns the partials into the five
// surface coefficients; PASS2 transforms the kept rows along x and stores the disc, detrended. PASS1 comes in two
// variants: one reads the sampler's grid, the fused one samples the frame itself (DESIGN section 12).
//
// The FFT is Stockham, radix 8 (4 or 2 for the last stage), eight points a thread in registers, one exchange through
// workgroup memory a stage. All stages read before any writes and write before any reads, so one buffer serves.
// ai: A size with a factor of 3 (384, 768, 1536) ends on a radix-3 stage, which runs in place (radix3Stage).
// Forward is exp(-i), unscaled inside a pass; 1 / n is applied at each pass's store, so Y = Z / n and S = T / n^2.
import { LUM0, MAP, SIZES, SLOTS } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { RING_COUNT } from "./finder.mjs";

// ai: The picture sizes and their count (common.mjs), a frame's picture slot its index in SIZES (sel.z, from F8's
// ai: version).
export { SIZES, SLOTS };
export const CPW = 4;   // picture columns a pass 1 workgroup: a quad is one 16 B load a row and one 16 B store of Y

// Stages: the product is n; radix 8 wherever it divides what is left.
// ai: The first is radix 8 (firstStage), and a factor of 3 is the last stage; any other n is refused.
export function stagesOf(n) {
  const s = [];
  for (let m = n; m > 1;) {
    const r = m % 8 === 0 ? 8 : m % 4 === 0 ? 4 : m % 2 === 0 ? 2 : m === 3 ? 3 : 0;
    if (!r || (s.length === 0 && r !== 8)) throw new Error(`no transform of ${n} points: 8 x 2^k or 8 x 3 x 2^k`);
    s.push(r); m /= r;
  }
  return s;
}

const FRAME = /* wgsl */ `
struct Frame { w: u32, h: u32, size: u32, valid: u32 }
`;
// ai: The per size uniform pass 1, reduce and pass 2 share (transform.mjs lane). p5 held the size's V until a frame's
// ai: rows came from LISTS (listsRows, 2026-09-26); p11 pads to 48 B.
const PARAMS = /* wgsl */ `
struct Params { gridStride: u32, yStride: u32, partStride: u32, size: u32, n: u32, p5: u32, m2: f32, sx2: f32, sq2: f32, sStride: u32, sampStride: u32, p11: u32 }
`;

// Per size: workgroups pass 1 takes a frame (n / CPW), the rows pass 2 takes (V), the blocks a frame (the slot B and C
// ai: may dispatch from), and 0 where the size is not built: a frame whose size index is SLOTS or more, or names an
// ai: unbuilt size, is left off every list. ARGS is u32 [SLOTS][12]: pass 1, reduce, pass 2, blocks; then from
// ai: argsLdpc pass one's LDPC dispatch (gpu/back/ldpc.mjs): (the most blocks of a size with a frame listed,
// ai: the most frames listed at one size, SLOTS), so a slot on no list (a frame with nothing found) and a size with no
// ai: frame dispatch no codeword's workgroup.
// ai: A frame's size is its picture slot, sel.z, which F8 sets from the version it decodes at (gpu/wgsl/word.mjs); a
// ai: frame is listed only with sel.y = 1 + that version (a found frame with no word and nothing held has sel.y 0).
// ai: V is the size's most rows; pass 2's slot of ARGS takes the rows its listed frames need (listsRows) instead.
// With zero set the gate also zeroes the batch's counters, verdicts, iteration counts and record count, so the
// composed chain needs no clearBuffer and can be appended to an open compute pass (back.mjs).
// ai: LISTS is u32 [SLOTS][1 + B], then (w, h) of every frame from word SLOTS (1 + B) (listsHead): the fused pass 1
// ai: clamps its bilinear reads to the frame's own edge, and taking the size from here keeps that pass at 8 storage
// ai: buffers.
// ai: Then from listsHead + 2 B each frame's block count (listsBlocks): the version F8 gave it (sel.y = 1 +
// ai: version, gpu/wgsl/word.mjs: its word's, or the held one) up to the size's blocks. Every version is a prefix of
// ai: its size's largest (src/focus.c: coefficients by frequency, the whitening by slot, block b on sub-channels 8b
// ai: to 8b + 7), so the soft stage and the LDPC take a frame's first count blocks and leave the rest.
// ai: Then from listsHead + 3 B each frame's row count (listsRows): the disc rows its first count blocks need
// ai: (gpu/back/transform.mjs rowsByCount, in the gate's uniform at gateRowsAt(count, size)), which pass 1's store and
// ai: pass 2 stop at. Pass 2's dispatch is the most rows any listed frame of the size needs; pass 1's and reduce's
// ai: stay a size's whole picture.
// ai: Then from listsHead + 4 B each frame's ring (listsRing, sel.x): the fused samplers (pass 1, the cancel
// ai: stage's fit) take the pair's grid and the ring's lattice from their PIC by it (PICTURE).
export const ARGS_WORDS = 12;
// ai: ARGS whole, u32 words: ARGS_WORDS a picture slot, then the LDPC's three.
export const argsLdpc = SLOTS * ARGS_WORDS;
export const argsWords = argsLdpc + 3;
export const listsHead = (B) => SLOTS * (1 + B);
export const listsWords = (B) => listsHead(B) + 5 * B;
export const listsBlocks = (B) => listsHead(B) + 2 * B;
export const listsRows = (B) => listsHead(B) + 3 * B;
export const listsRing = (B) => listsHead(B) + 4 * B;
export const ARGS_SLOT = { pass1: 0, reduce: 3, pass2: 6, blocks: 9 };
// ai: The gate's uniform in u32 words: at[s] = (n / CPW, V, blocks, 0) a slot, then the rows table flat, entry
// ai: c SLOTS + s the rows c blocks take at slot s (transform.mjs fills it), padded to whole vec4u. counts: block
// ai: counts 0 to the largest built size's blocks.
export const gateRowsAt = (c, s) => 4 * SLOTS + c * SLOTS + s;
export const gateWords = (counts) => 4 * SLOTS + 4 * Math.ceil((counts * SLOTS) / 4);
export function gateSource(B, { zero = false, blkStride = 0, counts } = {}) {
  if (!(counts > 1)) throw new Error(`gate: rows table of ${counts} block counts`);
  return FRAME + /* wgsl */ `
struct Gate { at: array<vec4u, ${SLOTS}>, rows: array<vec4u, ${Math.ceil((counts * SLOTS) / 4)}> }
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> sel: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> lists: array<u32>;
@group(0) @binding(3) var<storage, read_write> args: array<u32>;
@group(0) @binding(4) var<uniform> G: Gate;
${zero ? `@group(0) @binding(5) var<storage, read_write> counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> V: array<u32>;
@group(0) @binding(7) var<storage, read_write> ITS: array<u32>;
@group(0) @binding(8) var<storage, read_write> rec: array<u32>;` : ``}
const B: u32 = ${B}u;
const LS: u32 = ${B + 1}u;
const SLOTS: u32 = ${SLOTS}u;
const HEAD: u32 = ${listsHead(B)}u;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32) {
${zero ? `  for (var i: u32 = t; i < ${8 * B}u; i += 64u) { counts[i] = 0u; }
  for (var i: u32 = t; i < ${B * blkStride}u; i += 64u) { V[i] = 0u; ITS[i] = 0u; }
  if (t == 0u) { rec[0] = 0u; }` : ``}
  if (t != 0u) { return; }
  // ai: pass 2's workgroups a frame gather here as the frames are listed: the most rows any frame of the size needs.
  for (var s: u32 = 0u; s < SLOTS; s++) { lists[s * LS] = 0u; args[s * ${ARGS_WORDS}u + 6u] = 0u; }
  for (var f: u32 = 0u; f < B; f++) {
    lists[HEAD + 2u * f] = frames[f].w;
    lists[HEAD + 2u * f + 1u] = frames[f].h;
    lists[HEAD + 2u * B + f] = 0u;
    lists[HEAD + 3u * B + f] = 0u;
    lists[HEAD + 4u * B + f] = 0u;
    if (frames[f].valid == 0u || sel[f].y < 2u) { continue; }
    let s: u32 = sel[f].z;
    if (s >= SLOTS || G.at[s].x == 0u) { continue; }
    let c: u32 = lists[s * LS];
    lists[s * LS + 1u + c] = f;
    lists[s * LS] = c + 1u;
    let k: u32 = min(sel[f].y - 1u, G.at[s].z);
    let i: u32 = k * SLOTS + s;
    let r: u32 = G.rows[i >> 2u][i & 3u];
    lists[HEAD + 2u * B + f] = k;
    lists[HEAD + 3u * B + f] = r;
    lists[HEAD + 4u * B + f] = sel[f].x;
    args[s * ${ARGS_WORDS}u + 6u] = max(args[s * ${ARGS_WORDS}u + 6u], r);
  }
  var most: u32 = 0u;
  var wide: u32 = 0u;
  for (var s: u32 = 0u; s < SLOTS; s++) {
    let c: u32 = lists[s * LS];
    let a: u32 = s * ${ARGS_WORDS}u;
    args[a] = G.at[s].x; args[a + 1u] = 1u; args[a + 2u] = c;
    args[a + 3u] = 1u; args[a + 4u] = 1u; args[a + 5u] = c;
    args[a + 7u] = 1u; args[a + 8u] = c;
    args[a + 9u] = G.at[s].z; args[a + 10u] = 1u; args[a + 11u] = c;
    if (c > 0u) { most = max(most, c); wide = max(wide, G.at[s].z); }
  }
  args[${argsLdpc}u] = wide; args[${argsLdpc + 1}u] = most; args[${argsLdpc + 2}u] = SLOTS;
}
`;
}

// Complex helpers and the butterflies, on function-scope register arrays.
// ai: BUTTERFLIES, store, firstStage, laterStages and V8 are exported for the cancel stage's inverse transforms
// ai: (wgsl/cancel_paint.mjs): the same kernels, conj in and conj out.
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
// Twiddles (tw): "chain" reads the base twiddle from the table and gets the rest by multiplication (the default);
// "table" reads every twiddle; "sincos" computes the base one. All three are measured in test_transform.mjs.
export function laterStages(n, prec, twmode = "chain", lanes = [["v", "buf"]]) {
  const { ld, st } = store(prec);
  const T = n / 8, stages = stagesOf(n);
  let Ns = stages[0], out = "";
  for (let s = 1; s < stages.length; s++) {
    if (stages[s] === 3) { out += radix3Stage(n, prec, lanes); Ns *= 3; continue; }
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
      if (twmode === "sincos") out += `    let ang: f32 = -6.283185307179586 * f32(k) / ${Ns * R}.0;\n    let w1: vec2f = vec2f(cos(ang), sin(ang));\n`;
      else out += `    let w1: vec2f = tw[k * ${step}u];\n`;
      if (twmode === "table") { for (let r = 2; r < R; r++) out += `    let w${r}: vec2f = tw[k * ${r * step}u];\n`; }
      else {
        if (R >= 4) out += `    let w2: vec2f = cmul(w1, w1);\n    let w3: vec2f = cmul(w2, w1);\n`;
        if (R === 8) out += `    let w4: vec2f = cmul(w2, w2);\n    let w5: vec2f = cmul(w4, w1);\n    let w6: vec2f = cmul(w3, w3);\n    let w7: vec2f = cmul(w4, w3);\n`;
      }
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

// Pass 1 for one picture size: a workgroup takes CPW adjacent columns of one frame, four at a time. A quad's four
// columns are 16 contiguous bytes of every strip row, so they come in as one vec4f load a row (the scalar version
// touched the same 32 cache lines four times a row and pass 1 was 95% of the iGPU's time); the two pairs of a quad
// are transformed side by side, so Y(v) for the four columns leaves as one 16 B store (f16) or two (f32).
// ai: Params: gridStride (f32 a frame), yStride (complex a frame), partStride (vec4 a frame), the size index, n,
// ai: m2 = mean of cx^2, sx2, sq2, sStride and sampStride (vec2u a frame of the kept samples, transform.mjs).
// ai: Rows 1 to R - 1 of Y are stored, R the frame's rows from LISTS (listsRows): the rows past it hold no
// ai: coefficient of the blocks the frame decodes.
// ai: A column's mean rides in its PART record's w (pass 2 reads it for row v = 0), so pass 1 binds no Y0: that
// ai: binding is the one the cancel variant needs for REF under the 8 storage buffers a stage.
// grid: "f32" (the sampler's, one f32 an element) or "f16" (two f16 a u32, the same strip order and element index,
// half the bytes: what the sampler would write to halve the grid's traffic; the transform's arithmetic is unchanged).
// picture: "grid" reads the sampler's grid; "fused" samples the frame itself, F9's arithmetic a sample (the map, the
// Coons patch, the bilinear read; wgsl/sample.mjs), with a side's curve taken once a column or once a row. It binds
// ai: the frame texture, the maps and the residuals in place of the grid, and PIC (each ring's g0 and step for the
// ai: size, and the rings' lattices: PICTURE), taking the frame's ring from LISTS (listsRing).
export const GRID_FORMATS = ["f32", "f16"];
export const PICTURES = ["grid", "fused"];
export const LATTICE_MAX = 36;   // node coordinates a side PIC holds (register.mjs NODES_MAX / 4 is 34)
// ai: PIC, a picture size's uniform for the fused samplers: per ring (g0, step, nodes a side, first lattice
// ai: coordinate) as f32, the pair's grid (gpu/tables.mjs gridOf), then every ring's lattice, LATTICE_MAX a ring.
export const PICTURE = /* wgsl */ `
struct Picture { ring: array<vec4f, ${RING_COUNT}>, lat: array<vec4f, ${(RING_COUNT * LATTICE_MAX) / 4}> }
@group(0) @binding(9) var<uniform> PIC: Picture;
fn latAt(i: u32) -> f32 { return PIC.lat[i >> 2u][i & 3u]; }
`;
// ai: A fused sampler's head for frame f listed in `lists` (LISTS or LISTS2): its edge, map, ring's grid and lattice.
export const pictureHead = (lists, B) => `  let fw: i32 = i32(${lists}[${listsHead(B)}u + 2u * f]);\n  let fh: i32 = i32(${lists}[${listsHead(B)}u + 2u * f + 1u]);\n  let M: array<f32, 16> = maps[f];\n  let pr: vec4f = PIC.ring[${lists}[${listsRing(B)}u + f]];\n  let g0: f32 = pr.x;\n  let gstep: f32 = pr.y;\n  let nn: u32 = u32(pr.z);\n  let first: u32 = u32(pr.w);\n  let cn: Corners = corners(f, nn);\n`;
// Workgroup memory pass 1 declares for a size: the two exchange buffers and 48 B a thread of partials.
export const pass1WorkgroupBytes = (n, prec) => 2 * n * (prec === "f16" ? 4 : 8) + 48 * (n / 8);
// ai: keep: either picture variant also keeps every sample it read, as unorm16, in KEEP (binding 10, read_write:
// ai: quad-major, vec2u (four adjacent x) at f sampStride + (x / 4) n + y, so a workgroup's rows are one contiguous
// ai: store), for the cancel variant to read instead of sampling the frame again: pass two's sampling through
// ai: the map cost 0.56 of pass 1c's ms on the iGPU (DESIGN 12, 13.7). With it the fused pass binds 8 storage
// ai: buffers.
// ai: cancel: the cancel stage's variant (DESIGN 13, pass1c): the same pass over the second lists (LISTS2, ARGS2,
// ai: counting into BCOUNTS2), reading KEEP at binding 0 in place of a picture (picture is not consulted), with the
// ai: fitted mixture taken off each sample before the column sums, zz -= sum over the frame's references and their
// ai: three bases of (a + c v + d u) (b(x, y) - bbar(y)), the coefficients from the CANCEL uniform (binding 11:
// ai: { slotA, slotB, nrefs, ok, coef[18] } a frame, slot A's nine first, then slot B's; nothing is taken off a
// ai: frame whose solve failed) and the bases from REF (binding 10, read-only: a 24 B record a quad of columns a
// ai: row, the row means behind the slot's records; refSlot and refMeans in vec2u units).
export function pass1Source({ n, prec, B, cpw = CPW, tw = "chain", grid = "f32", picture = "grid", keep = false, cancel = false, refSlot = 0, refMeans = 0 }) {
  if (cpw % 4) throw new Error(`pass 1 takes columns four at a time: cpw ${cpw}`);
  if (!GRID_FORMATS.includes(grid)) throw new Error(`grid format ${grid}`);
  if (!PICTURES.includes(picture)) throw new Error(`pass 1 picture ${picture}`);
  if (cancel && !(refSlot > 0 && refMeans > 0)) throw new Error("the cancel variant needs REF's slot and means strides");
  if (cancel && keep) throw new Error("the cancel variant reads the kept samples; it keeps none");
  const { ty, ld } = store(prec);
  const T = n / 8, quads = cpw / 4, f16 = prec === "f16", g16 = grid === "f16", fused = !cancel && picture === "fused";
  const lanes = [["v0", "buf0"], ["v1", "buf1"]];
  // ai: Coefficient k of the frame's CANCEL record: 18 f32 behind the four header words, five vec4f; the solve
  // ai: (wgsl/cancel_fit.mjs) fills slot A's nine first, then slot B's, (1, v, u) a base, the rest zero.
  const cf = (k) => `cr.c${k >> 2}[${k & 3}]`;
  const model = (b, k0) => `(${cf(k0)} + ${cf(k0 + 1)} * yv + ${cf(k0 + 2)} * uu) * ${b}.d0 + (${cf(k0 + 3)} + ${cf(k0 + 4)} * yv + ${cf(k0 + 5)} * uu) * ${b}.d1 + (${cf(k0 + 6)} + ${cf(k0 + 7)} * yv + ${cf(k0 + 8)} * uu) * ${b}.d2`;
  let body = "";
  for (let q = 0; q < quads; q++) {
    body += `  {\n    let xq: u32 = x0 + ${4 * q}u;\n`;
    if (!fused && !cancel) body += `    let gq: u32 = (gbase + (xq >> 6u) * N * 64u + (xq & 63u)) >> 2u;\n`;
    // The top and bottom curves and the blend weight across depend on the column alone: once a column, not a sample.
    if (fused) for (let k = 0; k < 4; k++) body += `    let mx${k}: f32 = g0 + f32(xq + ${k}u) * gstep;\n    let u${k}: f32 = coonsT(mx${k}, first, nn);\n    let tp${k}: vec2f = curve(f, 0u, mx${k}, first, nn);\n    let bt${k}: vec2f = curve(f, 2u, mx${k}, first, nn);\n`;
    // ai: The quad's kept samples: one vec2u a row, rows contiguous.
    if (keep || cancel) body += `    let kq: u32 = kbase + (xq >> 2u) * N;\n`;
    // ai: The fit's u of the quad's four columns, x / (n - 1) - 0.5.
    if (cancel) body += `    let uu: vec4f = (vec4f(f32(xq)) + vec4f(0.0, 1.0, 2.0, 3.0)) / f32(N - 1u) - 0.5;\n`;
    body += `    var z: array<vec4f, 8> = array<vec4f, 8>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));\n`;
    body += `    var s1: vec4f = vec4f(0.0);\n    var sc: vec4f = vec4f(0.0);\n    var sq: vec4f = vec4f(0.0);\n`;
    for (let r = 0; r < 8; r++) {
      body += `    {\n      let y: u32 = tid + ${r * T}u;\n`;
      if (fused) {
        // The left and right curves and the weight down, once a row for the quad's four samples.
        body += `      let my: f32 = g0 + f32(y) * gstep;\n      let v: f32 = coonsT(my, first, nn);\n      let lf: vec2f = curve(f, 3u, my, first, nn);\n      let rg: vec2f = curve(f, 1u, my, first, nn);\n`;
        body += `      var zz: vec4f = vec4f(${[0, 1, 2, 3].map((k) => `sampleAt(f, M, vec2f(mx${k}, my), u${k}, v, tp${k}, bt${k}, lf, rg, cn, fw, fh)`).join(",\n        ")});\n`;
      } else if (cancel) body += `      let kw: vec2u = KEEP[kq + y];\n      var zz: vec4f = vec4f(unpack2x16unorm(kw.x), unpack2x16unorm(kw.y));\n`;
      else body += g16 ? `      let gw: vec2u = grid[gq + y * 16u];\n      var zz: vec4f = vec4f(unpack2x16float(gw.x), unpack2x16float(gw.y));\n` : `      var zz: vec4f = grid[gq + y * 16u];\n`;
      // ai: unorm16, not f16: a sample is luma in [0, 1] (the r8unorm texture, the C's lut), and unorm16 holds it to
      // ai: 7.6e-6 where f16 holds 2.4e-4 near 1 (f16 moved a block on test_cancel's batches, unorm16 none).
      if (keep) body += `      KEEP[kq + y] = vec2u(pack2x16unorm(zz.xy), pack2x16unorm(zz.zw));\n`;
      // ai: The subtraction at the sample: the frame's references' centred bases at this quad and row, weighted by
      // ai: the fitted planes (the archived arm's "full" model, one 24 B record a reference a row).
      if (cancel) {
        body += `      if (live) {\n        let yv: f32 = f32(y) / f32(N - 1u) - 0.5;\n        let bA: Bases = basesOf(slotA, xq, y);\n        zz -= ${model("bA", 0)};\n`;
        body += `        if (two) {\n          let bB: Bases = basesOf(slotB, xq, y);\n          zz -= ${model("bB", 9)};\n        }\n      }\n`;
      }
      body += `      z[${r}] = zz;\n`;
      // The detrend basis along y: c = (2 y + 1 - n) / n, q = c^2 less its mean (src/focus.c ws_init).
      body += `      let c: f32 = f32(2 * i32(y) + 1 - i32(N)) / f32(N);\n      s1 += zz;\n      sc += zz * c;\n      sq += zz * (c * c - P.m2);\n    }\n`;
    }
    // The column sums: eight lanes fold T / 8 entries each, then every lane folds the eight.
    body += `    red[3u * tid] = s1; red[3u * tid + 1u] = sc; red[3u * tid + 2u] = sq;\n    workgroupBarrier();\n`;
    body += `    if (tid < 8u) {\n      var a: vec4f = vec4f(0.0);\n      var b: vec4f = vec4f(0.0);\n      var c: vec4f = vec4f(0.0);\n`;
    body += `      for (var i: u32 = tid; i < T; i += 8u) { a += red[3u * i]; b += red[3u * i + 1u]; c += red[3u * i + 2u]; }\n`;
    body += `      red[3u * tid] = a; red[3u * tid + 1u] = b; red[3u * tid + 2u] = c;\n    }\n    workgroupBarrier();\n`;
    body += `    var S1: vec4f = vec4f(0.0);\n    var SC: vec4f = vec4f(0.0);\n    var SQ: vec4f = vec4f(0.0);\n`;
    body += `    for (var i: u32 = 0u; i < 8u; i++) { S1 += red[3u * i]; SC += red[3u * i + 1u]; SQ += red[3u * i + 2u]; }\n`;
    body += `    let mean: vec4f = S1 / f32(N);\n`;
    body += `    if (tid == 0u) {\n`;
    for (let k = 0; k < 4; k++) body += `      part[f * P.partStride + xq + ${k}u] = vec4f(S1[${k}], SC[${k}], SQ[${k}], mean[${k}]);\n`;
    body += `    }\n`;
    // Each column's mean comes out before the transform and travels on its own as Z(0) (DESIGN section 3).
    for (let r = 0; r < 8; r++) body += `    v0[${r}] = z[${r}].xy - mean.xy;\n    v1[${r}] = z[${r}].zw - mean.zw;\n`;
    body += firstStage(prec, lanes) + laterStages(n, prec, tw, lanes);
    // Separate each pair by conjugate symmetry (src/focus.c unpack_turn), scaled by 1 / n, rows 1 .. V - 1.
    // ai: V is the frame's own rows (listsRows), read at the head.
    body += `    for (var vv: u32 = 1u + tid; vv < rowsF; vv += T) {\n`;
    body += `      let z0: vec2f = ${ld("buf0", "vv")};\n      let m0: vec2f = ${ld("buf0", "N - vv")};\n      let z1: vec2f = ${ld("buf1", "vv")};\n      let m1: vec2f = ${ld("buf1", "N - vv")};\n`;
    body += `      let a0: vec2f = vec2f(z0.x + m0.x, z0.y - m0.y) * (0.5 / f32(N));\n      let b0: vec2f = vec2f(z0.y + m0.y, m0.x - z0.x) * (0.5 / f32(N));\n`;
    body += `      let a1: vec2f = vec2f(z1.x + m1.x, z1.y - m1.y) * (0.5 / f32(N));\n      let b1: vec2f = vec2f(z1.y + m1.y, m1.x - z1.x) * (0.5 / f32(N));\n`;
    body += `      let e: u32 = f * P.yStride + (vv - 1u) * N + xq;\n`;
    body += f16
      ? `      yb[e >> 2u] = vec4u(pack2x16float(a0), pack2x16float(b0), pack2x16float(a1), pack2x16float(b1));\n`
      : `      yb[e >> 1u] = vec4f(a0, b0);\n      yb[(e >> 1u) + 1u] = vec4f(a1, b1);\n`;
    body += `    }\n    workgroupBarrier();\n  }\n`;
  }
  // The fused variant's picture: the frame texture, its map, F7's residuals at the border nodes and, in PIC, where
  // ai: samples sit (g0, step) in each ring and the rings' lattices (PICTURE); bindings 1 to 6 are the grid variant's.
  const source = fused ? /* wgsl */ `
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(7) var<storage, read> maps: array<array<f32, 16>>;
@group(0) @binding(8) var<storage, read> resid: array<vec4f>;
` + PICTURE + MAP + LUM0 + COONS + /* wgsl */ `
fn sampleAt(f: u32, M: array<f32, 16>, m: vec2f, u: f32, v: f32, top: vec2f, bot: vec2f, lft: vec2f, rgt: vec2f, cn: Corners, fw: i32, fh: i32) -> f32 {
  return bilin0(f, toImage(M, m) + coonsMix(u, v, top, bot, lft, rgt, cn), fw, fh);
}
` : cancel ? /* wgsl */ `
@group(0) @binding(0) var<storage, read> KEEP: array<vec2u>;
` : /* wgsl */ `
@group(0) @binding(0) var<storage, read> grid: array<${g16 ? "vec2u" : "vec4f"}>;
`;
  // Per workgroup: the frame's size (from the gate), map and border corners (fused), or where its grid starts.
  // ai: With cancel, the frame's CANCEL record: its slots, whether the fit solved and its coefficients.
  const head = (fused ? pictureHead("lists", B) : cancel ? "" : `  let gbase: u32 = f * P.gridStride;\n`)
    + (keep || cancel ? `  let kbase: u32 = f * P.sampStride;\n` : "")
    + (cancel ? `  let cr: CancelRec = CANCEL[f];\n  let live: bool = cr.hdr.z > 0u && cr.hdr.w != 0u;\n  let two: bool = cr.hdr.z >= 2u;\n  let slotA: u32 = cr.hdr.x;\n  let slotB: u32 = cr.hdr.y;\n` : "");
  const keepSource = keep ? /* wgsl */ `
@group(0) @binding(10) var<storage, read_write> KEEP: array<vec2u>;
` : "";
  const cancelSource = cancel ? /* wgsl */ `
struct CancelRec { hdr: vec4u, c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, c4: vec4f }
@group(0) @binding(10) var<storage, read> REF: array<vec2u>;
@group(0) @binding(11) var<uniform> CANCEL: array<CancelRec, ${B}>;
const REF_SLOT: u32 = ${refSlot}u;
const REF_MEANS: u32 = ${refMeans}u;
struct Bases { d0: vec4f, d1: vec4f, d2: vec4f }
// ai: The three bases of a reference slot at the quad's four columns and row y, less the row's means: the record
// ai: 3 x vec4<f16> read as three vec2u, the means f32 behind the same view (wgsl/cancel_blur.mjs writes both).
fn basesOf(slot: u32, xq: u32, y: u32) -> Bases {
  let rec: u32 = REF_SLOT * slot + ((xq >> 2u) * N + y) * 3u;
  let w0: vec2u = REF[rec];
  let w1: vec2u = REF[rec + 1u];
  let w2: vec2u = REF[rec + 2u];
  let m: u32 = REF_SLOT * slot + REF_MEANS + 2u * y;
  let ma: vec2u = REF[m];
  let mb: vec2u = REF[m + 1u];
  return Bases(vec4f(unpack2x16float(w0.x), unpack2x16float(w0.y)) - bitcast<f32>(ma.x),
    vec4f(unpack2x16float(w1.x), unpack2x16float(w1.y)) - bitcast<f32>(ma.y),
    vec4f(unpack2x16float(w2.x), unpack2x16float(w2.y)) - bitcast<f32>(mb.x));
}
` : "";
  return (f16 ? "enable f16;\n" : "") + PARAMS + source + /* wgsl */ `
@group(0) @binding(1) var<storage, read> lists: array<u32>;
@group(0) @binding(2) var<storage, read> tw: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> part: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> yb: array<${f16 ? "vec4u" : "vec4f"}>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> P: Params;
const N: u32 = ${n}u;
const T: u32 = ${T}u;
const LS: u32 = ${B + 1}u;
const ROWS_AT: u32 = ${listsRows(B)}u;
var<workgroup> buf0: array<${ty}, N>;
var<workgroup> buf1: array<${ty}, N>;
var<workgroup> red: array<vec4f, ${3 * T}>;
` + keepSource + cancelSource + BUTTERFLIES + /* wgsl */ `
@compute @workgroup_size(T)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
  let f: u32 = lists[P.size * LS + 1u + wg.z];
  let rowsF: u32 = lists[ROWS_AT + f];
  let x0: u32 = wg.x * ${cpw}u;
  var v0: array<vec2f, 8> = ${V8};
  var v1: array<vec2f, 8> = ${V8};
` + head + body + /* wgsl */ `
  if (tid == 0u) { atomicAdd(&counts[f * 8u + 0u], ${cpw}u); }
}
`;
}

// The five surface coefficients from the column partials (src/focus.c focus_finish_bits, the px .. pyy lines).
export function reduceSource({ n, B }) {
  return PARAMS + /* wgsl */ `
@group(0) @binding(0) var<storage, read> lists: array<u32>;
@group(0) @binding(1) var<storage, read> part: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> coef: array<vec4f>;
@group(0) @binding(3) var<uniform> P: Params;
const N: u32 = ${n}u;
const LS: u32 = ${B + 1}u;
var<workgroup> red: array<vec4f, 512>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
  let f: u32 = lists[P.size * LS + 1u + wg.z];
  var s1: vec4f = vec4f(0.0);   // Sx, Sy, Sxy, Sxx
  var s2: f32 = 0.0;            // Syy
  for (var x: u32 = tid; x < N; x += 256u) {
    let p: vec4f = part[f * P.partStride + x];
    let c: f32 = f32(2 * i32(x) + 1 - i32(N)) / f32(N);
    let q: f32 = c * c - P.m2;
    s1 += vec4f(c * p.x, p.y, c * p.y, q * p.x);
    s2 += p.z;
  }
  red[tid] = s1;
  red[256u + tid] = vec4f(s2, 0.0, 0.0, 0.0);
  workgroupBarrier();
  if (tid < 32u) {
    var a: vec4f = vec4f(0.0);
    var b: vec4f = vec4f(0.0);
    for (var i: u32 = tid; i < 256u; i += 32u) { a += red[i]; b += red[256u + i]; }
    red[tid] = a;
    red[256u + tid] = b;
  }
  workgroupBarrier();
  if (tid == 0u) {
    var a: vec4f = vec4f(0.0);
    var b: vec4f = vec4f(0.0);
    for (var i: u32 = 0u; i < 32u; i++) { a += red[i]; b += red[256u + i]; }
    let nf: f32 = f32(N);
    coef[f * 2u] = vec4f(a.x / (nf * P.sx2), a.y / (nf * P.sx2), a.z / (P.sx2 * P.sx2), a.w / (nf * P.sq2));
    coef[f * 2u + 1u] = vec4f(b.x / (nf * P.sq2), 0.0, 0.0, 0.0);
  }
}
`;
}

// Pass 2 for one picture size: a workgroup takes row v of one frame (v = 0 from the column means, PART's w, in
// f32), transforms it along x and stores its compact disc row, the detrend subtracted in the C's units before the
// 1 / n^2.
export function pass2Source({ n, prec, B, tw = "chain" }) {
  const { ty, ld } = store(prec);
  const T = n / 8, f16 = prec === "f16";
  return (f16 ? "enable f16;\n" : "") + PARAMS + /* wgsl */ `
@group(0) @binding(0) var<storage, read> lists: array<u32>;
@group(0) @binding(1) var<storage, read> tw: array<vec2f>;
@group(0) @binding(2) var<storage, read> part: array<vec4f>;
@group(0) @binding(3) var<storage, read> yb: array<${ty}>;
@group(0) @binding(4) var<storage, read> coef: array<vec4f>;
@group(0) @binding(5) var<storage, read> x12: array<vec4f>;
// The disc rows as a uniform (V <= 512): pass 2 then binds 8 storage buffers, the default limit a phone may hold to.
struct Rows { r: array<vec4u, 512> }
@group(0) @binding(6) var<uniform> rows: Rows;
@group(0) @binding(7) var<storage, read_write> sb: array<${ty}>;
@group(0) @binding(8) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(9) var<uniform> P: Params;
const N: u32 = ${n}u;
const T: u32 = ${T}u;
const LS: u32 = ${B + 1}u;
const ROWS_AT: u32 = ${listsRows(B)}u;
var<workgroup> buf: array<${ty}, N>;
` + BUTTERFLIES + /* wgsl */ `
@compute @workgroup_size(T)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
  let f: u32 = lists[P.size * LS + 1u + wg.z];
  let vv: u32 = wg.x;
  // ai: A row past the frame's own (listsRows: its version's) holds no coefficient it decodes. Uniform: lists is
  // ai: read-only, so the return may precede the barriers.
  if (vv >= lists[ROWS_AT + f]) { return; }
  var v: array<vec2f, 8> = ${V8};
  if (vv == 0u) {
    for (var r: u32 = 0u; r < 8u; r++) { v[r] = vec2f(part[f * P.partStride + tid + r * T].w, 0.0); }
  } else {
    let ybase: u32 = f * P.yStride + (vv - 1u) * N;
    for (var r: u32 = 0u; r < 8u; r++) { v[r] = ${ld("yb", "ybase + tid + r * T")}; }
  }
` + firstStage(prec) + laterStages(n, prec, tw) + /* wgsl */ `
  let row: vec4u = rows.r[vv];
  let c0: vec4f = coef[f * 2u];
  let c1: vec4f = coef[f * 2u + 1u];
  let xv: vec4f = x12[vv];
  let nf: f32 = f32(N);
  let ustart: u32 = select(0u, 1u, vv == 0u);
  for (var e: u32 = tid; e < row.y; e += T) {
    var u: i32 = 0;
    if (e < row.z) { u = i32(e + ustart); } else { u = i32(e) - i32(row.z) - i32(row.w); }
    let m: u32 = u32(u + i32(N)) % N;
    let xu: vec4f = x12[m];
    // The C's spectral subtraction (src/focus.c focus_finish_bits lines 1001 to 1010) in its units: buf holds
    // T / n, so the surface's spectrum D is taken off as D / n, and S = (T - D) / n^2.
    var d: vec2f = c0.z * cmul(xu.xy, xv.xy);
    if (vv == 0u) { d += nf * (c0.x * xu.xy + c0.w * xu.zw); }
    if (u == 0) { d += nf * (c0.y * xv.xy + c1.x * xv.zw); }
    let s: vec2f = (${ld("buf", "m")} - d / nf) / nf;
    ${f16 ? `sb[f * P.sStride + row.x + e] = vec2<f16>(s);` : `sb[f * P.sStride + row.x + e] = s;`}
  }
  if (tid == 0u) {
    atomicAdd(&counts[f * 8u + 1u], 1u);
    atomicAdd(&counts[f * 8u + 2u], row.y);
  }
}
`;
}
