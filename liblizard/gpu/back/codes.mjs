// ai: The format's codes and rate profiles as the codec has them (2026-10-07, src/focus.h focus_tiers_for): what the
// ai: GPU decoder's rule stage (tiers.mjs) and the sender's GPU paint (../encoder.mjs sendTab, ../../gen/sender.mjs)
// ai: build their tables from, read from the wasm so the GPU never holds a code of its own.
import { init, Focus } from "../../sim/ob.mjs";
import { subchOf } from "../../sim/lizard_pick.mjs";
import { VERSIONS, CODES } from "../plan/back_tiers.mjs";
import { VERSION_MIN } from "../wordcode.mjs";

// ai: The codec's profiles and codes: profiles[v] = [{ c, firstBlock, slot0, count }] for v = VERSION_MIN .. VERSIONS
// ai: (the wasm's focus_tiers_for at 16 v, its tiers (rate, blocks, slots a block); profiles[0] null, no format; slot0
// ai: a tier's first slot of the frame, the blocks packed from slot 0), codes[c] in CODES' order (ldpcCode-shaped, from
// ai: a codec holding a block of each). The bit map is LINEAR, the format's one, in every table built from these.
export async function ruleTables() {
  const M = await init();
  const pT = M._malloc(4 * 3 * 4);   // ai: up to FOCUS_TIERS (4) tiers of (rate, blocks, slots a block)
  const profiles = Array(VERSION_MIN).fill(null);
  for (let v = VERSION_MIN; v <= VERSIONS; v++) {
    const n = M._focus_tiers_for(subchOf(v), pT);
    if (n < 1) throw new Error(`focus_tiers_for(${subchOf(v)}) gave no profile`);
    const t = M.HEAP32.slice(pT >> 2, (pT >> 2) + 3 * n);
    let firstBlock = 0, slot0 = 0;
    const tiers = [];
    for (let q = 0; q < n; q++) {
      const [rate, count, slots] = [t[3 * q], t[3 * q + 1], t[3 * q + 2]], c = CODES.findIndex((x) => x.rate === rate);
      if (c < 0 || CODES[c].slots !== slots) throw new Error(`LIZARD-${subchOf(v)}: a tier at rate ${rate} of ${slots} slots a block, which the GPU's codes lack`);
      tiers.push({ c, firstBlock, slot0, count });
      firstBlock += count; slot0 += count * slots;
    }
    if (slot0 > 640 * subchOf(v)) throw new Error(`LIZARD-${subchOf(v)}: its tiers take ${slot0} slots of ${640 * subchOf(v)}`);
    tiers.used = slot0;
    profiles.push(tiers);
  }
  M._free(pT);
  const fc = new Focus(256, 0, 1, { tiers: CODES.map(({ rate }) => [rate, 1]) });
  try {
    const pD = M._malloc(64);
    const codes = CODES.map((_, t) => {
      M._focus_ldpc_tables(t, pD, 0);
      const d = M.HEAP32.slice(pD >> 2, (pD >> 2) + 12);
      // ai: np: codeword bits never sent (the first np, the 7/8 code's first data column: 128), nt = n - np the bits a
      // ai: block's slots carry before its pilots (slots here: the code's lay entries, ldpc_tables's; a block's slots are CODES')
      const code = { n: d[0], k: d[1], m: d[2], z: d[3], zp: d[4], mb: d[5], kb: d[6], norm: d[7], slotsMax: d[8], slotsTotal: d[9], slots: d[10], np: d[11], nt: d[0] - d[11] };
      if (code.nt > CODES[t].slots || (CODES[t].slots - code.nt) % 2) throw new Error(`the ${CODES[t].name} code sends ${code.nt} bits, past its block's ${CODES[t].slots} slots`);
      const pLay = M._malloc(4 * (code.mb + 1 + 2 * code.slots));
      M._focus_ldpc_tables(t, pD, pLay);
      code.lay = M.HEAP32.slice(pLay >> 2, (pLay >> 2) + code.mb + 1 + 2 * code.slots);
      M._free(pLay);
      return code;
    });
    M._free(pD);
    return { profiles, codes };
  } finally { fc.free(); }
}

