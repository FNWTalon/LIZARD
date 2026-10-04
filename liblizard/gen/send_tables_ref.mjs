// ai: The GPU sender's tables as the web builds them (liblizard/gpu/encoder.mjs configureNow, shape, allocate), for
// ai: `tx_check gputables` to hold core/tx/send_tables.cpp to (2026-10-02). A format a file set in <out>:
// ai: <subch>_<ring>.<section>.bin, the sections rows, uv, tw, su, g, taps, border, pu, and index.json listing the
// ai: formats with their n, W, q, sq, Vr and copy. The formats: every whole number of blocks in the 128 ring (the
// ai: default) and a spread in the 32, 64 and 96, at 60 a second, encodes of 4 frames.
//   node gen/send_tables_ref.mjs <out> [all]      (all: every block count in every ring, 512 formats)
// ai:   CODES=2: two codes a frame side by side (gpu/encoder.mjs shape: FW, the gap, the paint's symbols FRAMES x codes)
import { mkdirSync, writeFileSync } from "node:fs";

const LIB = new URL("../", import.meta.url);
const imp = (p) => import(new URL(p, LIB).href);
const { init, Focus } = await imp("sim/ob.mjs");
const { transformTables } = await imp("gpu/back/transform.mjs");
const { PARAMS_AT } = await imp("gpu/wgsl/back_ldpc.mjs");
const { layWords } = await imp("gpu/back/ldpc.mjs");
const { crcPowers, PAYLOAD } = await imp("gpu/back/ref_ldpc.mjs");
const { clipLimit } = await imp("gpu/back/paint.mjs");
const { N_FOR, RINGS } = await imp("sim/lizard_pick.mjs");
await init();

const out = process.argv[2], all = process.argv[3] === "all";
if (!out) { console.error("usage: node gen/send_tables_ref.mjs <out> [all]"); process.exit(2); }
mkdirSync(out, { recursive: true });
const FRAMES = 4, FPS = 60, CODES = +(process.env.CODES ?? 1), GAP_MODULES = 12, formats = [];
for (let ring = 0; ring < 4; ring++) for (let b = 1; b <= 128; b++) if (all || ring === 3 || b % 9 === 1 || b === 128) formats.push({ subch: 8 * b, ring });
const index = [];
for (const { subch, ring } of formats) {
  const n = N_FOR(subch), span = 2 * RINGS[ring];
  const [tb] = await transformTables([{ n, subch }]);
  const fc = new Focus(n, subch, 1, { span, bitmap: tb.bitmap });
  fc.setFps(FPS);
  const g = fc.resampleGeom(), q = g.q, W = fc.side + 2 * fc.quiet * fc.cell, sq = g.o + fc.quiet * fc.cell;
  const gap = CODES > 1 ? GAP_MODULES * fc.cell : 0, FW = CODES * W + (CODES - 1) * gap, RW = Math.ceil(FW / 4), FS = 64 * Math.ceil((RW * W) / 64);
  const rows = new Uint32Array(2048); rows.set(tb.rows);
  const su = new ArrayBuffer(48), s32 = new Uint32Array(su), sf = new Float32Array(su), lim = clipLimit(subch);
  s32.set([n, tb.V, tb.V * n, tb.npos, (n * n) / 4, 0, 0, 0]); sf[8] = lim; sf[9] = Math.fround(0.5 / lim);
  const taps = new ArrayBuffer(48 * q), idx = new Uint32Array(taps, 0, 6 * q);
  for (let x = 0; x < q; x++) for (let t = 0; t < 6; t++) idx[6 * x + t] = (((g.i0[x] + t) % n) + n) % n;
  new Float32Array(taps, 24 * q, 6 * q).set(g.w);
  const copy = g.w.every((w, i) => w === (i % 6 === 2 ? 1 : 0));
  const G = new Uint32Array([n, q, sq, W, CODES, FW, RW, FS, 1, 0, gap, 0]);
  const pu = new Uint32Array(PARAMS_AT.words);
  pu.set([fc.blocks, 0, 0, 0], PARAMS_AT.sizes);
  pu.set([fc.blocks, tb.npos, FRAMES * CODES, 0], PARAMS_AT.dims);
  pu.set(layWords(tb.code), PARAMS_AT.lay);
  pu.set(crcPowers(PAYLOAD), PARAMS_AT.pw);
  const { rgba } = fc.encodeRGBA(new Uint8Array(fc.blocks * fc.blockBytes)), border = new Uint8Array(4 * Math.ceil((W * W) / 4));
  for (let i = 0; i < W * W; i++) border[i] = rgba[4 * i];
  fc.free();
  const name = `${subch}_${ring}`, put = (k, a) => writeFileSync(`${out}/${name}.${k}.bin`, new Uint8Array(a.buffer ?? a, a.byteOffset ?? 0, a.byteLength));
  put("rows", rows); put("uv", tb.UV); put("tw", tb.tw); put("su", new Uint8Array(su)); put("g", G); put("taps", new Uint8Array(taps)); put("border", border); put("pu", pu);
  index.push({ name, subch, ring, n, W, q, sq, Vr: tb.V, npos: tb.npos, copy, codes: CODES });
}
writeFileSync(`${out}/index.json`, JSON.stringify(index));
console.log(`${index.length} formats' tables in ${out}`);
