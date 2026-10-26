// The decoded blocks' lane buffers and their readback, which the rate profile's stage (tiers.mjs) writes: V (u32 a
// block: 0 not run, 1 verified, 2 declined, 3 stall or cap, 4 CRC), ITS (i32 a block, the C's codes) and REC (u32
// count, then 484-byte records: frame + 65536 block, iterations, 473 payload bytes, 3 pad). Every stride comes from
// dims.mjs derived().
// ai: Its own kernel, which decoded the one-rate format's 3/4 code, went 2026-10-22 (the profile's since 2026-10-07).
import { PAYLOAD } from "./ref_ldpc.mjs";
import { derived } from "./dims.mjs";

const U = globalThis.GPUBufferUsage;

export const REC_WORDS = 121;
export const REC_BYTES = 4 * REC_WORDS;

// ai: sizes[s] = { n, subch, blocks } at the size's index (sel[f].z), null where not served. REC holds every block of
// ai: every frame (B x blocksMax records).
export async function build(device, { B, sizes }) {
  const dims = derived(sizes, { B }), { served, blocksMax, recCap } = dims;
  const bytes = { V: 4 * B * blocksMax, ITS: 4 * B * blocksMax, REC: 4 + REC_BYTES * recCap };
  return {
    B, sizes, served, dims, blocksMax, recCap, bytes,
    // lane: adds V, ITS and REC.
    lane(ln) {
      ln.V = device.createBuffer({ size: bytes.V, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
      ln.ITS = device.createBuffer({ size: bytes.ITS, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
      ln.REC = device.createBuffer({ size: bytes.REC, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
      ln.bytes = (ln.bytes ?? 0) + bytes.V + bytes.ITS + bytes.REC;
      return ln;
    },
  };
}

// The records of one readback laid out a block: REC bytes (count first) to per frame verdict, iterations and
// payload arrays. Frames beyond those in the batch are ignored.
export function unpackRecords(recBytes, vWords, itsWords, { B, blocksMax, recCap }) {
  const dv = new DataView(recBytes.buffer, recBytes.byteOffset, recBytes.byteLength);
  const count = Math.min(dv.getUint32(0, true), recCap);
  const frames = [];
  for (let f = 0; f < B; f++) frames.push({ verdict: new Uint8Array(blocksMax), its: new Int8Array(blocksMax), bytes: new Uint8Array(blocksMax * PAYLOAD), records: 0 });
  for (let f = 0; f < B; f++) for (let b = 0; b < blocksMax; b++) { frames[f].verdict[b] = vWords[f * blocksMax + b]; frames[f].its[b] = itsWords[f * blocksMax + b]; }
  for (let r = 0; r < count; r++) {
    const o = 4 + REC_BYTES * r, tag = dv.getUint32(o, true), f = tag & 0xffff, b = tag >>> 16;
    if (f >= B || b >= blocksMax) continue;
    frames[f].bytes.set(new Uint8Array(recBytes.buffer, recBytes.byteOffset + o + 8, PAYLOAD), b * PAYLOAD);
    frames[f].records++;
  }
  return { count, frames };
}
