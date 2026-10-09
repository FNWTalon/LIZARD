// ai: The format's codes and rate profiles as the codec has them (2026-10-07, src/focus.h focus_tiers_for): what the
// ai: GPU decoder's rule stage (tiers.mjs) and the sender's GPU paint (../encoder.mjs sendTab, ../../gen/sender.mjs)
// ai: build their tables from, read from the wasm so the GPU never holds a code of its own.
import { init, Focus } from "../../sim/ob.mjs";
import { VERSIONS, CODES } from "../wgsl/back_tiers.mjs";

// ai: The codec's profiles and codes: profiles[v] = [{ c, firstBlock, sub0, count }] for v = 1 .. VERSIONS (the
// ai: wasm's focus_tiers_for at 8 v), codes[c] in CODES' order (ldpcCode-shaped, from a codec holding a block of
// ai: each), and the format's bit map mode.
export async function ruleTables() {
  const M = await init();
  const pT = M._malloc(4 * 3 * 4);   // ai: up to FOCUS_TIERS (4) tiers of (rate, blocks, sub-channels)
  const profiles = [null];
  for (let v = 1; v <= VERSIONS; v++) {
    const n = M._focus_tiers_for(8 * v, pT);
    if (n < 1) throw new Error(`focus_tiers_for(${8 * v}) gave no profile`);
    const t = M.HEAP32.slice(pT >> 2, (pT >> 2) + 3 * n);
    let firstBlock = 0, sub0 = 0;
    const tiers = [];
    for (let q = 0; q < n; q++) {
      const [rate, count, subs] = [t[3 * q], t[3 * q + 1], t[3 * q + 2]], c = CODES.findIndex((x) => x.rate === rate);
      if (c < 0 || CODES[c].subs !== subs) throw new Error(`LIZARD-${8 * v}: a tier at rate ${rate} of ${subs} sub-channels, which the GPU's codes lack`);
      tiers.push({ c, firstBlock, sub0, count });
      firstBlock += count; sub0 += count * subs;
    }
    if (sub0 !== 8 * v) throw new Error(`LIZARD-${8 * v}: its tiers fill ${sub0} sub-channels`);
    profiles.push(tiers);
  }
  M._free(pT);
  const fc = new Focus(256, 0, 1, { tiers: CODES.map(({ rate }) => [rate, 1]) });
  try {
    const bitmap = fc.bitmap, pD = M._malloc(64);
    const codes = CODES.map((_, t) => {
      M._focus_ldpc_tables(t, pD, 0);
      const d = M.HEAP32.slice(pD >> 2, (pD >> 2) + 12);
      // ai: np: codeword bits never sent (the first np, the 7/8 code's first data column: 93), nt = n - np the bits a block's slots carry
      const code = { n: d[0], k: d[1], m: d[2], z: d[3], zp: d[4], mb: d[5], kb: d[6], norm: d[7], slotsMax: d[8], slotsTotal: d[9], slots: d[10], np: d[11], nt: d[0] - d[11] };
      const pLay = M._malloc(4 * (code.mb + 1 + 2 * code.slots));
      M._focus_ldpc_tables(t, pD, pLay);
      code.lay = M.HEAP32.slice(pLay >> 2, (pLay >> 2) + code.mb + 1 + 2 * code.slots);
      M._free(pLay);
      return code;
    });
    M._free(pD);
    return { profiles, codes, bitmap };
  } finally { fc.free(); }
}

