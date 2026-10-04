// ai: The SPEC.md conformance vectors (5.2 through the painted cells, 6.12 P2 and P3, 7.4, 9.3) against one build,
// ai: plus digests over the whole ladder (layout cells at five rates, the drive, the coefficient order) to diff
// ai: between two builds. A build from before the finder frame's deletion takes a thin argument; this passes 1 there.
//   node test/vectors.mjs <path/to/ob.mjs>      from liblizard/, about 20 s
import { createHash } from "node:crypto";
import { resolve } from "node:path";

const path = process.argv[2] ?? "build/ob.mjs";
const M = await (await import(resolve(path))).default();
const sha = (u8) => createHash("sha256").update(u8).digest("hex");
const hex = (u8) => Buffer.from(u8).toString("hex");
let fails = 0;
const want = (name, got, exp) => { const ok = got === exp; if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${got}${ok ? "" : `  (SPEC ${exp})`}`); };

const oldApi = M._focus_setup.length === 13;
// ai: span 2B names ring B, 0 the default ring (the 128 since 2026-10-01). SPEC 6.12's vectors and the ladder's digests
// ai: are the 64 ring's, so it is named: span 128.
const setup = (n, subch, span = 128) => (oldApi ? M._focus_setup(n, subch, 1, 2, span, 1, 0, 0, 0, 0, 0, 0, 0) : M._focus_setup(n, subch, 1, 2, span, 0, 0, 0, 0, 0, 0, 0));
// ai: The tree's ladder (2^k and 3 x 2^k since 2026-09-27): a build from before it cannot build the 3 x 2^k sizes.
const { N_FOR: nFor } = await import(resolve("sim/lizard_pick.mjs"));
console.log(`${path}: focus_setup takes ${M._focus_setup.length} arguments`);

// ai: (int)(v * 255 + 0.5) clamped, in float32 as focus_paint_rgba's scalar path does it (SPEC 6.8, Painting).
const grey = (v) => { const t = Math.fround(Math.fround(v * 255) + 0.5); const i = Math.trunc(t); return i < 0 ? 0 : i > 255 ? 255 : i; };

// ai: again = false encodes on the codec already set up, whose block bytes the ladder loop holds.
let blockBytes = 0;
function encode(n, subch, fill, again = true) {
  const B = again ? setup(n, subch) : blockBytes;
  if (B < 0) throw new Error(`setup ${n}/${subch} refused`);
  const T = M._focus_blocks(), side = M._focus_side();
  if (M._focus_dbg(1)) throw new Error("dbg");
  const pB = M._malloc(T * B), pD = M._malloc(4 * side * side);
  const blocks = new Uint8Array(T * B);
  fill(blocks, B, T);
  M.HEAPU8.set(blocks, pB);
  M._focus_tx(pB, pD);
  const drive = M.HEAPF32.slice(pD >> 2, (pD >> 2) + side * side);
  const sym = new Int8Array(M.HEAPU8.buffer, M._focus_dbg_sym(), 2 * 320 * subch).slice();
  M._focus_dbg(0); M._free(pB); M._free(pD);
  return { B, T, side, drive, slots: Uint8Array.from(sym, (s) => (s < 0 ? 1 : 0)) };
}
const pack = (bits) => { const o = new Uint8Array(Math.ceil(bits.length / 8)); bits.forEach((b, i) => { if (b) o[i >> 3] |= 0x80 >> (i & 7); }); return o; };
const greyRow = (e, y, x0, k) => Array.from({ length: k }, (_, i) => grey(e.drive[y * e.side + x0 + i])).join(" ");
const greySquare = (e, a, b) => { const o = new Uint8Array((b - a) * (b - a)); let q = 0; for (let y = a; y < b; y++) for (let x = a; x < b; x++) o[q++] = grey(e.drive[y * e.side + x]); return o; };

// ai: 5.2's words, each symbol painted in the ring named, the word read back out of the layout's cells: word cell c
// ai: of side s is the c-th CELL_WORD cell along the band (depth 5), dark = 1, and shows bit 7 - c mod 8 of byte
// ai: 4 floor(c / 8) + s (5.3).
{
  const pDims = M._malloc(24);
  for (const [subch, B, fps, exp] of [
    [16, 32, 0, "4c0200723f148394"],
    [512, 64, 24, "4c401843c236ee95dc072a1c61cb63fc"],
    [1024, 64, 240, "4c80f01a03b06a272bf125d409be9dd9"],
    [576, 64, 30, "4c481e8a5889b2ffdc13581b53702083"],
    [16, 64, 0, "4c0200bf8a272653d9ff5f11927947ed"],
    [256, 96, 120, "4c20782d28b496c17b9c80f73a2b385db0ea9d6286a4bf2e"],
    [64, 128, 60, "4c083c787c7c416d2b6b3fc18075004ad5cd92b1945ab6165f18ed021c053ff6"],
  ]) {
    if (setup(nFor(subch), subch, 2 * B) < 0 || M._focus_fmt_fps_set(fps)) throw new Error(`setup LIZARD-${subch} ring ${B}`);
    M._ob_test_mesh_tables_out(pDims, 0, 0, 0, 0);
    const d = M.HEAP32.slice(pDims >> 2, (pDims >> 2) + 6), S = d[2];
    const pX = M._malloc(4 * d[0]), pY = M._malloc(4 * d[1]), pK = M._malloc(S * S), pMark = M._malloc(4 * d[0] * d[1]);
    M._ob_test_mesh_tables_out(pDims, pX, pY, pK, pMark);
    const kind = M.HEAPU8.slice(pK, pK + S * S);
    [pX, pY, pK, pMark].forEach((p) => M._free(p));
    const at = [(t) => [t, 5], (t) => [S - 6, t], (t) => [t, S - 6], (t) => [5, t]], sides = [];
    for (let s = 0; s < 4; s++) {
      const bits = [];
      for (let t = 0; t < S; t++) { const [x, y] = at[s](t), k = kind[y * S + x]; if (k & 8) { bits.push((k & 3) === 2 ? 1 : 0); t++; } }
      sides.push(bits);
    }
    const cw = new Uint8Array(exp.length / 2);
    for (let k = 0; k < cw.length; k++) for (let b = 0; b < 8; b++) cw[k] |= (sides[k % 4][8 * Math.floor(k / 4) + b] ?? 0) << (7 - b);
    want(`5.2 LIZARD-${subch} ring ${B} fps ${fps} (${sides[0].length} word cells a side)`, hex(cw), exp);
  }
  M._free(pDims);
}
// ai: 6.12 P2, LIZARD-16, every payload byte zero: n 256 in the 64 ring (the default from 2026-09-27 to 10-01): side 316,
// ai: pxm 2, the picture from pixel 30, the resampled square [24, 292) (modules 12 to 145: the picture and its
// ai: 3-module guard, FOCUS_CP = 3).
{
  const e = encode(256, 16, () => {});
  want("P2 slot bits sha256", sha(e.slots), "726c1e022ebb8c61cd6ded139be3a029b0102d39f81fdceeb08c70410edea142");
  want("P2 packed first 16", hex(pack(e.slots).subarray(0, 16)), "00203e010efc0be0f8bfbdce000c0c03");
  want("P2 drive side", String(e.side), "316");
  want("P2 drive row 30 cols 30..33", Array.from(e.drive.subarray(30 * 316 + 30, 30 * 316 + 34), (v) => v.toFixed(7)).join(", "), "0.7166191, 1.0000000, 1.0000000, 1.0000000");
  want("P2 grey row 30 cols 30..45", greyRow(e, 30, 30, 16), "183 255 255 255 255 177 121 120 138 131 96 57 41 61 104 135");
  want("P2 grey row 230 cols 230..245", greyRow(e, 230, 230, 16), "255 255 119 22 9 42 74 89 93 104 132 176 211 213 182 148");
  const sq = greySquare(e, 24, 292);
  want("P2 square sha256", sha(sq), "4b0fece66fb3a1963ae8d322893800afedcd2164914f6c519b75fc2033b76116");
  want("P2 square zeros / 255s", `${sq.filter((v) => v === 0).length} / ${sq.filter((v) => v === 255).length}`, "1619 / 1698");
  want("P2 whole drive sha256", sha(greySquare(e, 0, 316)), "49f87e1569ffd094642136cb1025ab20e038f23b8d068a0f2ce774e851ff377a");
}
// ai: 6.12 P3, LIZARD-64, every payload byte zero: n 384 in the 64 ring, side 474, pxm 3 (a copy), the picture
// ai: from pixel 45, the resampled square [36, 438).
{
  const e = encode(nFor(64), 64, () => {});
  want("P3 slot bits sha256", sha(e.slots), "530e72f118c1c3f3b90b6017548c13039b3c9415922ff234a6a10853af28250e");
  want("P3 drive side", String(e.side), "474");
  want("P3 grey row 45 cols 45..60", greyRow(e, 45, 45, 16), "94 159 255 255 196 145 208 229 141 71 91 129 132 127 138 156");
  want("P3 square sha256", sha(greySquare(e, 36, 438)), "3da487dd12bb2fd61466373bb6377ff80a1afa2deae35d4ddc21dc36db074a1a");
  want("P3 whole drive sha256", sha(greySquare(e, 0, 474)), "ec6c4e4cd91acb3aaaac44e9ab13ce99078a25aed4784d7baa60daaf0ad0593f");
}
// ai: 7.4, LIZARD-16, block b = id b LE32 then byte j = j mod 256.
{
  const e = encode(256, 16, (blocks, B) => { for (let b = 0; b < 2; b++) { blocks[b * B] = b; for (let j = 0; j < 469; j++) blocks[b * B + 4 + j] = j & 255; } });
  const blk = (b) => { const u = new Uint8Array(473); u[0] = b; for (let j = 0; j < 469; j++) u[4 + j] = j & 255; return u; };
  for (const b of [0, 1]) {
    const p = M._malloc(473); M.HEAPU8.set(blk(b), p);
    want(`7.4 block ${b} CRC-32`, "0x" + (M._ob_test_crc_hash(p, 473) >>> 0).toString(16).toUpperCase().padStart(8, "0"), b ? "0x90C7A3FE" : "0x9F59BCE3");
    M._free(p);
    want(`7.4 block ${b} slots 0..127`, hex(pack(e.slots.subarray(5120 * b, 5120 * b + 128))), b ? "77df2df63b06611785ae191b9d491b47" : "2a6d2d409722af16f5f2c5ec83cda6fb");
    want(`7.4 block ${b} slots 5088..5119`, hex(pack(e.slots.subarray(5120 * b + 5088, 5120 * b + 5120))), b ? "8afe77aa" : "a909ab10");
  }
  want("7.4 frame slots packed sha256", sha(pack(e.slots)), "8881113e5edfc7b2c3ac85c58d00b57d4186021e9c3a66a045715263e95cbd1f");
}
// ai: 9.3, the test stream.
{
  const p = M._malloc(469);
  for (const [id, exp] of [[0, "b4a21e7939beb125bd3c10237ed5ea31"], [1, "220ee668c9ea457d0330c21554da9e11"], [12345, "08781f021b171adcc9fd8886013dffc2"], [4294967295, "8c82cb0926f0275dc7650c348eae46b2"]]) {
    M._stream_fill(id >>> 0, p, 469);
    want(`9.3 id ${id}`, hex(M.HEAPU8.subarray(p, p + 16)), exp);
  }
  M._stream_fill(0, p, 469);
  want("9.3 id 0 tail", hex(M.HEAPU8.subarray(p + 461, p + 469)), "2149dd1e8e99a4fc");
  M._free(p);
}
// ai: The whole ladder, to diff between builds: every format's cell map (word cells included) at the rates 5.2's
// ai: vectors state, its drive on a fixed payload, and its coefficient order.
{
  const cells = createHash("sha256"), drives = createHash("sha256"), order = createHash("sha256");
  const pDims = M._malloc(24), pX = M._malloc(4 * 64), pY = M._malloc(4 * 64), pMark = M._malloc(4 * 64 * 64);
  for (let subch = 16; subch <= 1024; subch += 16) {
    const n = nFor(subch);
    if ((blockBytes = setup(n, subch)) < 0) throw new Error(`setup ${n}/${subch}`);
    for (const fps of [24, 30, 60, 240, 0]) {
      if (M._focus_fmt_fps_set(fps)) throw new Error("fps");
      M._ob_test_mesh_tables_out(pDims, 0, 0, 0, 0);
      const d = M.HEAP32.subarray(pDims >> 2, (pDims >> 2) + 6), w = d[2], h = d[3];
      const pK = M._malloc(w * h);
      M._ob_test_mesh_tables_out(pDims, pX, pY, pK, pMark);
      const dd = Array.from(M.HEAP32.subarray(pDims >> 2, (pDims >> 2) + 6));
      cells.update(JSON.stringify([subch, fps, dd])).update(M.HEAPU8.subarray(pK, pK + w * h))
        .update(M.HEAPU8.subarray(pX, pX + 4 * dd[0])).update(M.HEAPU8.subarray(pY, pY + 4 * dd[1])).update(M.HEAPU8.subarray(pMark, pMark + 4 * dd[0] * dd[1]));
      M._free(pK);
    }
    const e = encode(n, subch, (blocks) => { for (let k = 0; k < blocks.length; k++) blocks[k] = (k * 37 + subch) & 255; }, false);
    drives.update(new Uint8Array(e.drive.buffer));
    const pp = M._focus_pos_ptr();
    order.update(M.HEAPU8.subarray(pp, pp + 4 * 320 * subch));
  }
  console.log(`ladder cells ${cells.digest("hex").slice(0, 16)}  drives ${drives.digest("hex").slice(0, 16)}  order ${order.digest("hex").slice(0, 16)}`);
}
console.log(fails ? `${fails} FAILED` : "all SPEC vectors ok");
process.exit(fails ? 1 : 0);
