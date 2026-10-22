// The per batch strides every stage of the back half sizes its buffers by, from one function, so the soft-value
// stage (B), the LDPC (C) and the composition (back.mjs) cannot disagree on where frame f's slots sit. Before this
// each stage recomputed them from its own view of the size list (DESIGN.md section 11).
//
// ai: sizes[s] = { n, subch, blocks, ... } at the size's index (sel[f].z, the picture slot), null where the size is not
// ai: built. A size with no blocks is not served.
export function derived(sizes, { B, cap = 0 } = {}) {
  const served = sizes.map((z, s) => (z && z.blocks > 0 ? s : -1)).filter((s) => s >= 0);
  if (!served.length) throw new Error("no size served");
  const subchMax = Math.max(...served.map((s) => sizes[s].subch));
  const blocksMax = Math.max(...served.map((s) => sizes[s].blocks));
  const nmax = Math.max(...served.map((s) => sizes[s].n));
  return {
    B, served, subchMax, blocksMax, nmax,
    sStride: Math.max(...served.map((s) => sizes[s].subch * 320)),   // S entries a frame (the largest disc)
    lStride: subchMax * 160,                                          // L words a frame: int8 slot order, 640 a sub-channel
    estStride: subchMax,                                              // EST entries a frame
    blkStride: blocksMax,                                             // BLK, V and ITS entries a frame
    recCap: cap > 0 ? cap : (B ?? 1) * blocksMax,                     // records the result buffer holds a batch
  };
}
