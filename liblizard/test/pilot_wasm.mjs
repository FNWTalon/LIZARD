// ai: The pilots at version 1 (SPEC 7.3; 2026-10-07): a one-block symbol carries bit 0 of the painted count alone, so
// ai: a decode reads r and no odd reading: the told C (Focus) and the blind C the receiver runs (FocusAny) report r2
// ai: and its error as NaN, never a zero, which a lock would take for a reading of an even mix. At version 2 and up
// ai: both readings are there. A clean capture painted at count c reads r = +1 at c = 0 and -1 at c = 1 (bit 0), r2
// ai: +1 at c = 0 and 2, -1 at c = 1 and 3 (bit 1); |r| >= 0.9 is the bound scripts/exp/pilot_check.mjs holds a
// ai: clean capture to. The symbol is painted by the codec at a pixel a drive pixel on white, no camera.
//   node test/pilot_wasm.mjs      from liblizard/, a few seconds; OB_BUILD=<path> for another build; exit 1 on a failure
import { init, Focus, FocusAny } from "../sim/ob.mjs";
import { N_FOR } from "../sim/lizard_pick.mjs";

await init();
let fails = 0;
const want = (name, ok, got) => { if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${got}`); };
const P = 24;   // ai: white round the symbol
const fa = new FocusAny(1536);
for (const subch of [8, 16]) {
  const V = subch / 8, n = N_FOR(subch), fc = new Focus(n, subch, 1);
  for (const c of [0, 1, 2, 3]) {
    const blocks = new Uint8Array(fc.blocks * fc.blockBytes);
    for (let i = 0; i < blocks.length; i++) blocks[i] = ((i + 7919 * (c + 1)) * 2654435761) >>> 13 & 255;
    fc.setParity(c);
    const { w, rgba } = fc.encodeRGBA(blocks), W = w + 2 * P, img = new Uint8Array(W * W).fill(255);
    for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) img[(y + P) * W + x + P] = rgba[4 * (y * w + x)];
    const s0 = c & 1 ? -1 : 1, s1 = c & 2 ? -1 : 1;
    for (const [who, p] of [["told", fc.decode(img, W, W).pilot], ["blind", fa.decode(img, W, W).pilot]]) {
      const tag = `LIZARD-${subch} c=${c} ${who}`;
      want(`${tag} blocks`, !!p && p.blocks === V, p ? p.blocks : "null");
      if (!p) continue;
      want(`${tag} r`, Math.sign(p.r) === s0 && Math.abs(p.r) >= 0.9 && Number.isFinite(p.sd), `${p.r.toFixed(3)} sd ${p.sd.toFixed(4)}`);
      if (V === 1) want(`${tag} r2 none`, Number.isNaN(p.r2) && Number.isNaN(p.sd2), `${p.r2} sd ${p.sd2}`);
      else want(`${tag} r2`, Math.sign(p.r2) === s1 && Math.abs(p.r2) >= 0.9 && Number.isFinite(p.sd2), `${p.r2.toFixed(3)} sd ${p.sd2.toFixed(4)}`);
    }
  }
  fc.free();
}
console.log(fails ? `FAILED: ${fails}` : "all ok");
process.exit(fails ? 1 : 0);
