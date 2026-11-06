// ai: The picture slots and the frame record's axis word: what every decoder stage, the back half's tables and the
// ai: sender's paint share (2026-10-31: from glsl/common.mjs, whose WGSL pieces went with the WGSL decoder, archive/webgpu-decoder/).

import { PICTURE_SIZES, VERSIONS, N_FOR } from "../../sim/lizard_pick.mjs";

// ai: The picture sizes a format has (sim/lizard_pick.mjs PICTURE_SIZES, those N_FOR gives a version of the ladder: all
// ai: seven, 384 to 1536; gpu/tables.mjs PICTURES), a frame's picture slot its
// ai: index here (sel.z, from F8's version), and SLOTS of them. Every per-slot table on the device (F8's tops, F9's
// ai: sizes, the back half's gate, LISTS, ARGS, the LDPC's and the paint's uniforms, gate2's) is SLOTS long, so a
// ai: ladder change edits no stage. Here, not in back_transform.mjs, because sample.mjs needs it and back_transform.mjs
// ai: imports sample.mjs.
export const SIZES = PICTURE_SIZES.filter((n) => VERSIONS.some((s) => N_FOR(s) === n));
export const SLOTS = SIZES.length;

// ai: The axis word: 0 the frame's centre (a centred crop's axis), else x and y in half pixels, each + 32768 in 16
// ai: bits (x low): a 2:1 half's axis is the middle of its inner edge, (w, h / 2) on the left, (0, h / 2) on the right
// ai: (core/rx/receiver.cpp).
export const AXIS_CENTRE = 0;
export function axisWord(axis) {
  if (!axis) return AXIS_CENTRE;
  const h = (v) => Math.max(0, Math.min(65535, Math.round(2 * v) + 32768));
  return (h(axis[0]) | (h(axis[1]) << 16)) >>> 0;
}
