// The sender's paint uniform layout (send.mjs) and the split CRC-32, shared by the rate profile's LDPC kernel
// (back_tiers.mjs) and the sender's paint.
// ai: The one-rate LDPC kernel these came with (ldpcSource: layered min-sum on the 3/4 code, the stall rule and the
// ai: learned stop) went 2026-10-22; the profile's kernel is back_tiers.mjs tldpcSource.
// The CRC-32 is split over lanes (ref_ldpc.mjs crcPowers): lane w runs the reflected bitwise CRC over its 4-byte
// chunk from a zero state, multiplies by x^(8 * bytes after it) mod P from a table, and XORs into one word; lane 0
// adds the initial state's term with the final inversion folded in. 119 lanes, 32 steps each, no table in
// workgroup memory.
import { SLOTS } from "./common.mjs";

// ai: The sender's paint uniform P in u32 words (wgsl/send.mjs paintSource): sizes, a vec4u a picture slot; dims; pw,
// ai: 30 vec4u (the CRC-32 powers, crcPowers(PAYLOAD)). PARAMS_STRUCT is its WGSL.
export const PARAMS_AT = { sizes: 0, dims: 4 * SLOTS, pw: 4 * SLOTS + 4, words: 4 * SLOTS + 124 };
export const PARAMS_STRUCT = `struct Params { sizes: array<vec4<u32>, ${SLOTS}>, dims: vec4<u32>, pw: array<vec4<u32>, 30> }`;

// ai: The split CRC-32 as WGSL, shared with the sender's paint (wgsl/send.mjs): crcOf(i) is called by
// ai: every lane of the workgroup with its lane index (it holds two barriers) once the 473 payload bytes sit in the
// ai: workgroup array `packw` (120 u32, four bytes a word in memory order, byte 0 of word 118 the last) and returns
// ai: the CRC to every lane. `crcw` names the workgroup atomic it folds into, `crcStep` the name the step function
// ai: is declared under, `pw` the powers table expression (120 u32: crcPowers(PAYLOAD), as the LDPC's uniform holds
// ai: it). The names are parameters so a shader with its own arrays binds them without a copy of the arithmetic.
export function CRC_SPLIT({ packw = "packw", crcw = "crcw", crcStep = "crcStep", pw = "P.pw" } = {}) {
  return /* wgsl */ `
const CRC_CHUNKS: u32 = 119u;
const CRC_POLY: u32 = 0xedb88320u;
fn ${crcStep}(c: u32) -> u32 { return (c >> 1u) ^ (CRC_POLY & (0u - (c & 1u))); }
fn crcOf(i: u32) -> u32 {
  if (i == 0u) { atomicStore(&${crcw}, 0u); }
  workgroupBarrier();
  if (i < CRC_CHUNKS) {
    let w = ${packw}[i];
    var nb: u32 = 4u;
    if (i == CRC_CHUNKS - 1u) { nb = 1u; }
    var c: u32 = 0u;
    for (var t: u32 = 0u; t < nb; t = t + 1u) {
      c = c ^ ((w >> (8u * t)) & 0xffu);
      for (var q: u32 = 0u; q < 8u; q = q + 1u) { c = ${crcStep}(c); }
    }
    let p = ${pw}[i >> 2u][i & 3u];
    var acc: u32 = 0u;
    var tt = c;
    for (var j: u32 = 0u; j < 32u; j = j + 1u) {
      if (((p >> (31u - j)) & 1u) != 0u) { acc = acc ^ tt; }
      tt = ${crcStep}(tt);
    }
    if (i == 0u) { acc = acc ^ ${pw}[CRC_CHUNKS >> 2u][CRC_CHUNKS & 3u]; }
    atomicXor(&${crcw}, acc);
  }
  workgroupBarrier();
  return atomicLoad(&${crcw});
}
`;
}
