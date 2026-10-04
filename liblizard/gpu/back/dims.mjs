// The per batch strides every stage of the back half sizes its buffers by, from one function, so the soft-value
// stage (B), the LDPC (C) and the composition (back.mjs) cannot disagree on where frame f's slots sit. Before this
// each stage recomputed them from its own view of the size list (DESIGN.md section 11).
//
// ai: sizes[s] = { n, subch, blocks, ... } at the size's index (sel[f].z, the picture slot), null where the size is not
// ai: built. A size with no blocks is not served.
// ai: The cancel stage's strides too (DESIGN.md section 13): refSlots (R) paint slots a batch, half the frames plus
// ai: one up to 16 (4 with a picture past 1024 built: a REF slot is 14 MB at 1536), or the count given; refStride,
// ai: bytes a REF slot (bases, 24 B a quad of columns a row over nmax, then the row means, vec4f a row, at
// ai: refMeans); refBytes, REF whole (R slots and the carry slot R); picStride, bytes a PIC slot (f16 a sample);
// ai: fitpStride, bytes a frame of FITP (the fit's workgroups at nmax, FITP_WORDS f32 each); bitmaps, a size's bit
// ai: map mode, for PERMW.
import { FITP_WORDS, fitWorkgroups } from "../wgsl/cancel_fit.mjs";
export function derived(sizes, { B, cap = 0, refSlots = 0 } = {}) {
  const served = sizes.map((z, s) => (z && z.blocks > 0 ? s : -1)).filter((s) => s >= 0);
  if (!served.length) throw new Error("no size served");
  const subchMax = Math.max(...served.map((s) => sizes[s].subch));
  const blocksMax = Math.max(...served.map((s) => sizes[s].blocks));
  const nmax = Math.max(...served.map((s) => sizes[s].n));
  const R = refSlots > 0 ? refSlots : Math.min(Math.floor((B ?? 1) / 2) + 1, nmax > 1024 ? 4 : 16);
  const refStride = 6 * nmax * nmax + 16 * nmax;
  return {
    B, served, subchMax, blocksMax, nmax,
    sStride: Math.max(...served.map((s) => sizes[s].subch * 320)),   // S entries a frame (the largest disc)
    lStride: subchMax * 160,                                          // L words a frame: int8 slot order, 640 a sub-channel
    estStride: subchMax,                                              // EST entries a frame
    blkStride: blocksMax,                                             // BLK, V and ITS entries a frame
    recCap: cap > 0 ? cap : (B ?? 1) * blocksMax,                     // records the result buffer holds a batch
    refSlots: R, refStride, refMeans: 6 * nmax * nmax, refBytes: (R + 1) * refStride,
    picStride: 2 * nmax * nmax, fitpStride: 4 * FITP_WORDS * fitWorkgroups(nmax),
    bitmaps: sizes.map((z) => (z ? z.bitmap | 0 : 0)),
  };
}
