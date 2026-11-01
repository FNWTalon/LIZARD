// ai: The SPEC.md conformance vectors (5.2 through the painted cells, 6.12 P2 and P3, 7.4, 9.3) against one build,
// ai: plus digests over the ladder (layout cells at five rates, the drive, the coefficient order) to diff between two
// ai: builds of today's interface (focus_setup's five arguments since 2026-10-25).
//   node test/vectors.mjs [path/to/ob.mjs]      from liblizard/, about 20 s; default build/ob.mjs, the codec's module
import { createHash } from "node:crypto";
import { resolve } from "node:path";

const path = process.argv[2] ?? "build/ob.mjs";
const M = await (await import(resolve(path))).default();
// ai: a block and its payload, bytes (src/xfer.h XFER_BLOCK, XFER_PAYLOAD)
const BLOCK = 476, PAYLOAD = BLOCK - 4;
const sha = (u8) => createHash("sha256").update(u8).digest("hex");
const hex = (u8) => Buffer.from(u8).toString("hex");
let fails = 0;
const want = (name, got, exp) => { const ok = got === exp; if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${got}${ok ? "" : `  (SPEC ${exp})`}`); };

// ai: span 2B names ring B, 0 the default ring (the 128 since 2026-10-01). SPEC 6.12's vectors and the ladder's digests
// ai: are the 64 ring's, so it is named: span 128.
const setup = (n, subch, span = 128) => M._focus_setup(n, subch, 2, span, 0);
// ai: The tree's ladder (2^k and 3 x 2^k since 2026-09-27, 1152 and 1280 since 2026-10-29): a build from before it
// ai: cannot build the sizes added since.
const { N_FOR: nFor, VERSIONS } = await import(resolve("sim/lizard_pick.mjs"));
console.log(`${path}: focus_setup takes ${M._focus_setup.length} arguments`);

// ai: (int)(v * 255 + 0.5) clamped, in float32 as focus_paint_rgba's scalar path does it (SPEC 6.8, Painting).
const grey = (v) => { const t = Math.fround(Math.fround(v * 255) + 0.5); const i = Math.trunc(t); return i < 0 ? 0 : i > 255 ? 255 : i; };

// ai: again = false encodes on the codec already set up, whose block bytes the ladder loop holds.
let blockBytes = 0;
function encode(n, subch, fill, again = true, span = 128) {
  const B = again ? setup(n, subch, span) : blockBytes;
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
    [512, 64, 24, "4c20188a931bcb154e849dfc68bc499e"],
    [1024, 64, 240, "4c40f095a1ea203a12ea56091b50c91d"],
    [576, 64, 30, "4c241ef7ae82b06f1b6a9de7f66d48aa"],
    [64, 64, 0, "4c0400e55734bb5b7d82661fc44c6646"],
    [64, 128, 60, "4c043c7618ef9352e525c51045b722df9e8064f9f0ddad60a761d030c9963206"],
    [256, 256, 120, "4c10783d4504e4341b703313d38614a0a2d894f9ca7e94d881959561fec187689d9017a57e59ffeb675e0b2d074a080b48ab41f8c3a9c9cdc88dea6bd98c991c"],
  ]) {
    if (setup(nFor(subch), subch, 2 * B) < 0 || M._focus_fmt_fps_set(fps)) throw new Error(`setup LIZARD-${subch} ring ${B}`);
    M._ob_test_mesh_tables_out(pDims, 0, 0, 0, 0);
    const d = M.HEAP32.slice(pDims >> 2, (pDims >> 2) + 5), S = d[2];
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
// ai: 6.12 P2, LIZARD-64, every payload byte zero: n 384 in the 128 ring: side 572, pxm 2, the picture resampled by 4/3
// ai: from pixel 30, the resampled square [24, 548) (modules 12 to 273: the picture and its 3-module guard,
// ai: FOCUS_CP = 3), the anchors (SPEC 6.13) 4 modules from modules 31, 68, 104, 141, 178, 214, 251.
// ai: (LIZARD-32 at n 256 in the 64 ring until 2026-11-01, LIZARD-16 until 2026-10-31.)
{
  const e = encode(384, 64, () => {}, true, 256);
  want("P2 slot bits sha256", sha(e.slots), "04e27f794b080a1ebc30d409405851e011e18ac904c713fa75f9bb6917799353");
  want("P2 packed first 16", hex(pack(e.slots).subarray(0, 16)), "00043e000efd03e0f0ffffcc000c0c03");
  want("P2 drive side", String(e.side), "572");
  want("P2 drive row 30 cols 30..33", Array.from(e.drive.subarray(30 * 572 + 30, 30 * 572 + 34), (v) => v.toFixed(7)).join(", "), "0.4950773, 0.4724883, 0.6975312, 0.9888784");
  want("P2 grey row 30 cols 30..45", greyRow(e, 30, 30, 16), "126 120 178 252 255 209 139 131 188 242 231 160 91 65 89 129");
  want("P2 grey row 345 cols 345..360", greyRow(e, 345, 345, 16), "113 94 95 99 110 140 183 217 207 151 75 18 5 35 118 211");
  want("P2 grey row 64 cols 60..71 (anchor 0, 0)", greyRow(e, 64, 60, 12), "62 111 255 255 0 0 0 0 255 255 106 114");
  const sq = greySquare(e, 24, 548);
  want("P2 square sha256", sha(sq), "8aeda0414b3b27d7711a5acd994531a16c14acffb8f0a4f025cb65a59d4de4e3");
  want("P2 square zeros / 255s", `${sq.filter((v) => v === 0).length} / ${sq.filter((v) => v === 255).length}`, "4897 / 6900");
  want("P2 whole drive sha256", sha(greySquare(e, 0, 572)), "27c1a4bdd0794e7b955a94a7336f78a51e1186b0a38b17e772ec05d8b12ddb77");
}
// ai: 6.12 P3, LIZARD-64, every payload byte zero: n 384 in the 64 ring, side 474, pxm 3 (a copy), the picture
// ai: from pixel 45, the resampled square [36, 438), the anchors as P2's.
{
  const e = encode(nFor(64), 64, () => {});
  want("P3 slot bits sha256", sha(e.slots), "04e27f794b080a1ebc30d409405851e011e18ac904c713fa75f9bb6917799353");
  want("P3 packed first 16 (P2's)", hex(pack(e.slots).subarray(0, 16)), "00043e000efd03e0f0ffffcc000c0c03");
  want("P3 drive side", String(e.side), "474");
  want("P3 grey row 45 cols 45..60", greyRow(e, 45, 45, 16), "136 154 247 255 163 126 206 249 176 91 81 119 148 139 109 122");
  want("P3 grey row 71 cols 66..77 (anchor 0, 0)", greyRow(e, 71, 66, 12), "62 96 126 255 0 255 255 0 255 86 96 91");
  want("P3 square sha256", sha(greySquare(e, 36, 438)), "35967d2409d3eff8c40d94b1e449bd6a70c1bbf432d532c6074c6600f7209f3b");
  want("P3 whole drive sha256", sha(greySquare(e, 0, 474)), "db6422590e50b169cfee2561e925bec21be730286e32b3e68f11168aecf7a1a6");
}
// ai: 7.4, LIZARD-64 (7/8 x 4, 3/4, 2/3 and 1/2 blocks from frame slots 0, 4496, 8992, 13,488, 17,984, 23,136 and
// ai: 28,896, the padding from 36,576 to 40,960; LIZARD-32 until 2026-11-01, LIZARD-16 until 2026-10-31), block b = id b
// ai: LE32 then byte j = j mod 256: each block's CRC-32, its first 128 slots, its tail (the pilots: a 7/8 block's 16,
// ai: the 3/4 block's 32; the 2/3 and 1/2 blocks have none) and the padding's first.
{
  const e = encode(384, 64, (blocks, B) => { for (let b = 0; b < 7; b++) { blocks[b * B] = b; for (let j = 0; j < PAYLOAD; j++) blocks[b * B + 4 + j] = j & 255; } });
  const blk = (b) => { const u = new Uint8Array(BLOCK); u[0] = b; for (let j = 0; j < PAYLOAD; j++) u[4 + j] = j & 255; return u; };
  const WANT = [
    { at: 0, nt: 4480, slots: 4496, crc: "0x4AB56DA5", head: "75327e236757d37a0084f2c064a9db97", tail: "5a69" },
    { at: 4496, nt: 4480, slots: 4496, crc: "0x84A6ED54", head: "1518eac8960db08464eec512ada2cc43", tail: "6e57" },
    { at: 8992, nt: 4480, slots: 4496, crc: "0x0DE36A06", head: "de7f05a7aa40e9efa256a13154f265e1", tail: "80c1" },
    { at: 13488, nt: 4480, slots: 4496, crc: "0xC3F0EAF7", head: "3003b3c57ffdd161de5670c9f1bfbdfc", tail: "bc79" },
    { at: 17984, nt: 5120, slots: 5152, crc: "0xC41962E3", head: "0603a4229de65af270a8f2db1111d193", tail: "c22fe70f" },
    { at: 23136, nt: 5760, slots: 5760, crc: "0x0A0AE212", head: "a90b8ae3e08dd639660698828a536799", tail: "" },
    { at: 28896, nt: 7680, slots: 7680, crc: "0x834F6540", head: "060e92c7b025451292c7ec2e384e367b", tail: "" },
  ];
  WANT.forEach((w, b) => {
    const p = M._malloc(BLOCK); M.HEAPU8.set(blk(b), p);
    want(`7.4 block ${b} CRC-32`, "0x" + (M._ob_test_crc_hash(p, BLOCK) >>> 0).toString(16).toUpperCase().padStart(8, "0"), w.crc);
    M._free(p);
    want(`7.4 block ${b} slots 0..127 (frame slots ${w.at}..)`, hex(pack(e.slots.subarray(w.at, w.at + 128))), w.head);
    if (w.slots > w.nt) want(`7.4 block ${b} tail slots ${w.nt}..${w.slots - 1}`, hex(pack(e.slots.subarray(w.at + w.nt, w.at + w.slots))), w.tail);
  });
  want("7.4 padding slots 36576..36703", hex(pack(e.slots.subarray(36576, 36704))), "332c1ead5f960d5aaf4c1e8d9f9e7cd8");
  want("7.4 frame slots packed sha256", sha(pack(e.slots)), "994f4750a6596def85a66fdbcacb9866edbbc827c1c408ad6ba5e010793b46cb");
}
// ai: 9.3, the test stream.
{
  const p = M._malloc(PAYLOAD);
  for (const [id, exp] of [[0, "b4a21e7939beb125bd3c10237ed5ea31"], [1, "220ee668c9ea457d0330c21554da9e11"], [12345, "08781f021b171adcc9fd8886013dffc2"], [4294967295, "8c82cb0926f0275dc7650c348eae46b2"]]) {
    M._stream_fill(id >>> 0, p, PAYLOAD);
    want(`9.3 id ${id}`, hex(M.HEAPU8.subarray(p, p + 16)), exp);
  }
  M._stream_fill(0, p, PAYLOAD);
  want("9.3 id 0 tail", hex(M.HEAPU8.subarray(p + PAYLOAD - 8, p + PAYLOAD)), "1e8e99a4fc614f4e");
  M._free(p);
}
// ai: The whole ladder, to diff between builds: every format's cell map (word cells included) at the rates 5.2's
// ai: vectors state, its drive on a fixed payload, and its coefficient order.
{
  const cells = createHash("sha256"), drives = createHash("sha256"), order = createHash("sha256");
  const pDims = M._malloc(24), pX = M._malloc(4 * 64), pY = M._malloc(4 * 64), pMark = M._malloc(4 * 64 * 64);
  for (const subch of VERSIONS) {
    const n = nFor(subch);
    if ((blockBytes = setup(n, subch)) < 0) throw new Error(`setup ${n}/${subch}`);
    for (const fps of [24, 30, 60, 240, 0]) {
      if (M._focus_fmt_fps_set(fps)) throw new Error("fps");
      M._ob_test_mesh_tables_out(pDims, 0, 0, 0, 0);
      const d = M.HEAP32.subarray(pDims >> 2, (pDims >> 2) + 5), w = d[2], h = d[3];
      const pK = M._malloc(w * h);
      M._ob_test_mesh_tables_out(pDims, pX, pY, pK, pMark);
      // ai: nx, ny, w, h, corner (the layout's thin flag, always 1, sat before the corner until 2026-10-25)
      const dd = Array.from(M.HEAP32.subarray(pDims >> 2, (pDims >> 2) + 5));
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
