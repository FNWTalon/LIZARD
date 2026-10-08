// ai: The app's GPU sender (2026-10-02; the CPU already held 60 a second at LIZARD-1024): the web's encoder kernels
// ai: (liblizard/gpu/encoder.mjs: paint, irows and ipic a picture size, tpose, rsv, rsh and its copy form) as the WGSL
// ai: the web builds them from, for gen.mjs to compile as it does the decoder's, and the tables no format changes,
// ai: which the web host computes from the wasm and core/tx/gpu_painter.cpp takes from here: the paint's TAB (every
// ai: code of the format's rate profile, src/focus.h focus_tiers_for, and the whitening: gpu/encoder.mjs sendTab, since
// ai: 2026-10-07; one code's bit map a block, PERMW, before) and the CRC's powers. What a format does change (the disc's
// ai: rows, the resampler's taps, the border, its tiers in the paint's uniform) the painter takes from the C's codec on
// ai: the phone (core/tx/send_tables.cpp), held to the web's by gen/send_tables_ref.mjs and `tx_check gputables`.
import { createHash } from "node:crypto";

const LIB = new URL("../", import.meta.url);
const imp = (p) => import(new URL(p, LIB).href);

// ai: { modules: { name: WGSL }, manifest (setup/send.json), tab: TAB's bytes (blobs/<hash>.bin) }
export async function senderSetup() {
  const { irowsSource, ipicSource } = await imp("gpu/wgsl/cancel_paint.mjs");
  const { paintSource, tposeSource, rsvSource, rshSource, TPOSE_TILE, RSV_THREADS, RSH_THREADS } = await imp("gpu/wgsl/send.mjs");
  const { SLOTS, SIZES } = await imp("gpu/wgsl/common.mjs");
  const { PARAMS_AT } = await imp("gpu/wgsl/back_ldpc.mjs");
  const { CODES } = await imp("gpu/wgsl/back_tiers.mjs");
  const { crcPowers, PAYLOAD } = await imp("gpu/back/ref_ldpc.mjs");
  const { CLIP } = await imp("gpu/back/paint.mjs");
  const { ruleTables } = await imp("gpu/back/codes.mjs");
  const { GAP_MODULES, sendTab } = await imp("gpu/encoder.mjs");
  const modules = { paint: paintSource(), tpose: tposeSource(), rsv: rsvSource(), rsh: rshSource(), rshCopy: rshSource({ copy: true }) };
  for (const n of SIZES) {
    modules[`irows${n}`] = irowsSource({ n, prec: "f32", mode: "send" });
    modules[`ipic${n}`] = ipicSource({ n, prec: "f32", mode: "send" });
  }
  // ai: the codes from the codec (the wasm's, as the web's encoder reads them), in CODES' order: the paint's uniform
  // ai: names a tier by its place there
  const rt = await ruleTables(), tab = sendTab(rt);
  if (SLOTS < 1 + CODES.length) throw new Error(`the paint's uniform holds ${SLOTS} sizes, the codes' tiers need ${1 + CODES.length}`);
  const tabBytes = new Uint8Array(tab.buffer, tab.byteOffset, tab.byteLength);
  const tabHash = createHash("sha256").update(tabBytes).digest("hex").slice(0, 16);
  const manifest = {
    modules: Object.keys(modules).map((k) => `send_${k}`), sizes: SIZES, slots: SLOTS, paramsAt: PARAMS_AT,
    tposeTile: TPOSE_TILE, rsvThreads: RSV_THREADS, rshThreads: RSH_THREADS, blockBytes: 480, payload: PAYLOAD, clip: CLIP, gapModules: GAP_MODULES,
    bitmap: rt.bitmap, codes: CODES.map(({ rate, subs }) => ({ rate, subs })), pw: Array.from(crcPowers(PAYLOAD)), tab: { blob: `blobs/${tabHash}.bin`, words: tab.length },
  };
  return { modules, manifest, tab: { hash: tabHash, bytes: tabBytes } };
}
