// The soft values' tables and lane buffers, which the rate profile's stage (tiers.mjs) decodes through: the UV
// table a size (transform.mjs discLayout) with each coefficient's (u, v) behind it, the per-size params
// (wgsl/back_tiers.mjs Params), and a lane's L (int8 slot order, 4 a u32, lStride words a frame), BLK (a block:
// bitcast f32 estimate, u32 declined) and PILOT. Every stride comes from dims.mjs derived().
// ai: Its own soft and align kernels, which decoded one code rate, went 2026-10-22 (the profile's since 2026-10-07).
import { derived } from "./dims.mjs";

const U = globalThis.GPUBufferUsage;   // undefined under node

// sizes[s] = { n, subch, blocks } (null for a size not served); tables[s] = { UV, UC: Uint32Array subch * 320 }
// (transform.mjs transformTables).
export async function build(device, { B, sizes, tables }) {
  const dims = derived(sizes, { B }), { served, subchMax, blocksMax, sStride, lStride } = dims;
  for (const s of served) if (!tables[s]?.UV || !tables[s]?.UC) throw new Error(`size ${s} (${sizes[s].n}/${sizes[s].subch}) has no UV and UC tables`);
  // The UV tables, one buffer, each size at its own entry offset.
  // ai: and after them all, at the same offsets again, each coefficient's (u, v) (transform.mjs discLayout UC) for
  // ai: the turn by the grid's shift.
  let uvTotal = 0;
  const uvOff = [];
  for (const s of served) { uvOff[s] = uvTotal; uvTotal += tables[s].UV.length; }
  const uvBuf = device.createBuffer({ size: 8 * uvTotal, usage: U.STORAGE | U.COPY_DST });
  for (const s of served) {
    device.queue.writeBuffer(uvBuf, 4 * uvOff[s], tables[s].UV);
    device.queue.writeBuffer(uvBuf, 4 * (uvTotal + uvOff[s]), tables[s].UC);
  }
  // ai: Params a size (wgsl/back_tiers.mjs Params): the slot, UV's offset, the strides, the batch size, UC's offset
  // ai: and n, in a 48-byte uniform.
  const params = [];
  for (const s of served) {
    const z = sizes[s], buf = device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST });
    const u32 = new Uint32Array(12);
    u32.set([s, uvOff[s], sStride, lStride, subchMax, blocksMax, B, uvTotal + uvOff[s], z.n]);
    device.queue.writeBuffer(buf, 0, u32);
    params[s] = buf;
  }
  // ai: PILOT: a block's pilots, blocksMax a frame, then a frame's shift, two each (talign)
  const bytes = { L: 4 * B * lStride, BLK: 8 * B * blocksMax, PILOT: 4 * B * blocksMax + 8 * B };
  return {
    B, sizes, served, dims, sStride, lStride, subchMax, blocksMax, bytes, uvBuf, uvOff, uvTotal, params,
    // lane: { S, LISTS, BCOUNTS } from part A. Adds L, BLK and PILOT.
    lane(ln) {
      ln.L = device.createBuffer({ size: bytes.L, usage: U.STORAGE | U.COPY_SRC });
      ln.BLK = device.createBuffer({ size: bytes.BLK, usage: U.STORAGE | U.COPY_SRC });
      ln.PILOT = device.createBuffer({ size: bytes.PILOT, usage: U.STORAGE | U.COPY_SRC });
      ln.bytes = (ln.bytes ?? 0) + bytes.L + bytes.BLK + bytes.PILOT;
      return ln;
    },
  };
}
