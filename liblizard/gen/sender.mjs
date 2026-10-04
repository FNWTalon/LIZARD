// ai: The app's GPU sender (2026-10-02; the CPU already held 60 a second at LIZARD-1024): the web's encoder kernels
// ai: (liblizard/gpu/encoder.mjs: paint, irows and ipic a picture size, tpose, rsv, rsh and its copy form) as the WGSL
// ai: the web builds them from, for gen.mjs to compile as it does the decoder's, and the tables no format changes,
// ai: which the web host computes from the wasm and core/tx/gpu_painter.cpp takes from here: the LDPC's lay words (one
// ai: code for every format), the CRC's powers, and the bit map's row a block (PERMW: block b's row is the same in
// ai: every format, so a format of V blocks reads the first V rows of 128; asserted below). What a format does change
// ai: (the disc's rows, the resampler's taps, the border) the painter takes from the C's codec on the phone
// ai: (core/tx/send_tables.cpp), held to the web's by gen/send_tables_ref.mjs and `tx_check gputables`.
import { createHash } from "node:crypto";

const LIB = new URL("../", import.meta.url);
const imp = (p) => import(new URL(p, LIB).href);

// ai: { modules: { name: WGSL }, manifest (setup/send.json), perm: the PERMW bytes (blobs/<hash>.bin) }
export async function senderSetup() {
  const { paintSource, irowsSource, ipicSource } = await imp("gpu/wgsl/cancel_paint.mjs");
  const { tposeSource, rsvSource, rshSource, TPOSE_TILE, RSV_THREADS, RSH_THREADS } = await imp("gpu/wgsl/send.mjs");
  const { SLOTS, SIZES } = await imp("gpu/wgsl/common.mjs");
  const { PARAMS_AT } = await imp("gpu/wgsl/back_ldpc.mjs");
  const { layWords, packMap } = await imp("gpu/back/ldpc.mjs");
  const { crcPowers, PAYLOAD } = await imp("gpu/back/ref_ldpc.mjs");
  const { permTable } = await imp("gpu/back/bitmap.mjs");
  const { CLIP } = await imp("gpu/back/paint.mjs");
  const { GAP_MODULES } = await imp("gpu/encoder.mjs");
  const { init, Focus } = await imp("sim/ob.mjs");
  await init();
  const modules = { paint: paintSource({ prec: "f32", mode: "send" }), tpose: tposeSource(), rsv: rsvSource(), rsh: rshSource(), rshCopy: rshSource({ copy: true }) };
  for (const n of SIZES) {
    modules[`irows${n}`] = irowsSource({ n, prec: "f32", mode: "send" });
    modules[`ipic${n}`] = ipicSource({ n, prec: "f32", mode: "send" });
  }
  // ai: the code and the bit map from the largest format and the smallest, which must agree (one code, one bit map)
  const big = new Focus(1536, 1024, 1), small = new Focus(256, 8, 1);
  const code = big.ldpcCode().code, bitmap = big.bitmap, BLOCKS = big.blocks;
  const lay = layWords(code), lay8 = layWords(small.ldpcCode().code);
  if (small.bitmap !== bitmap || lay.some((w, i) => w !== lay8[i])) throw new Error("the code or the bit map differs between formats: the sender's tables assume one");
  big.free(); small.free();
  const perm = permTable({ blocks: BLOCKS, mode: bitmap });
  for (const V of [1, 7, 60, 127]) {
    const p = permTable({ blocks: V, mode: bitmap });
    for (let i = 0; i < p.length; i++) if (p[i] !== perm[i]) throw new Error(`PERMW of ${V} blocks is not the first rows of ${BLOCKS}'s (entry ${i})`);
  }
  const permBytes = new Uint8Array(packMap(perm).buffer);
  const permHash = createHash("sha256").update(permBytes).digest("hex").slice(0, 16);
  const manifest = {
    modules: Object.keys(modules).map((k) => `send_${k}`), sizes: SIZES, slots: SLOTS, paramsAt: PARAMS_AT,
    tposeTile: TPOSE_TILE, rsvThreads: RSV_THREADS, rshThreads: RSH_THREADS, blockBytes: 480, payload: PAYLOAD, clip: CLIP, gapModules: GAP_MODULES,
    bitmap, lay: Array.from(lay), pw: Array.from(crcPowers(PAYLOAD)), perm: { blob: `blobs/${permHash}.bin`, blocks: BLOCKS, slotsPerBlock: perm.length / BLOCKS },
  };
  return { modules, manifest, perm: { hash: permHash, bytes: permBytes } };
}
