// ai: The clipping level at the format's clip ratio c = 2 (SPEC.md 6.7; src/lizard_phy.h LIZARD_CLIP): the C's f32
// ai: arithmetic (lizard_phy.c lizard_encode), lim = 2 c sqrt(0.5 subch 320). The GPU encoder (../encoder.mjs) and the
// ai: native sender's tables (gen/send_tables_ref.mjs) take it from here.
export const CLIP = 2;
export const clipLimit = (subch) => Math.fround(Math.fround(2 * CLIP) * Math.fround(Math.sqrt(Math.fround(0.5 * subch * 320))));
