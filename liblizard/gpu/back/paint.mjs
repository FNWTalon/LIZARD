// ai: The clip every sender paints at today (SPEC.md 13 item 8): the C's f32 arithmetic (focus.c focus_encode), lim =
// ai: 2 clip sqrt(0.5 subch 320) at tilt 0. The GPU encoder (../encoder.mjs) and the native sender's tables
// ai: (gen/sender.mjs, gen/send_tables_ref.mjs) take it from here.
export const CLIP = 2;
export const clipLimit = (subch, clip = CLIP) => Math.fround(Math.fround(2 * clip) * Math.fround(Math.sqrt(Math.fround(0.5 * subch * 320))));
