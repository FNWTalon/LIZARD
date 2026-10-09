// Which LIZARD code to show, from the room available and nothing else. No DOM: lizard-web/send.mjs
// reads its controls and calls this, scripts/exp/pick_check.mjs exercises it exhaustively.
//
// A VERSION IS ITS SUB-CHANNEL COUNT and nothing else, and the natural unit is the BLOCK of 8 sub-channels
// (src/focus.h FOCUS_GROUP): the codec rejects a count that is not a multiple of 8. So version k = 8k sub-channels;
// the blocks a frame are the rate profile's (tiersFor below), k of them to LIZARD-56 and at most 7 under k above,
// each 469 B of payload.
//
// THE FIRST LADDER (2026-09-20, one code rate; the history of the decision, not the rule: VERSIONS below is).
// Versions 2 to 32, so 16 to 256 sub-channels in steps of 8. The version number IS the block count, which is worth
// keeping even though it means the ladder starts at 2: version 12 says 12 blocks and 5628 B without a lookup.
// The top is 256 and not the 408 the picture could hold because everything above it only wins on a clean 1080 or
// better (research/10 s13: 400 carries 21 574 B a frame at 1080 but 2609 at 480, where 128 carries 5716).
// The bottom is 16 and not 8 because a single-block frame has no diversity at all: one failed block is an empty
// frame. Nothing measured wants fewer than 32 in any case (32 was never optimal in the 13 cells, and the worst of
// them, 2.5 px defocus, wanted 64), so versions 2 and 3 are already provision for conditions worse than any
// simulated.
//
// Steps of 8 and not 32: an earlier version of this used 32 to make eight rungs fit three bits of the format word,
// which does not hold up, because that word has to carry n, the rate and the mode as well, so it needs about eight
// bits either way and will live in a block where five bits against three costs nothing. The coarse ladder had a
// real price: the pick rounds DOWN to a rung, so a rule asking for 120 took 96 and gave up 24 sub-channels, 1407 B
// a frame, a quarter of the payload. Steps of 8 cut the worst rounding loss from 31 sub-channels to 7.
//
// The picture size n is DERIVED (N_FOR): it decides only how finely the same coefficients are sampled. It is
// derived from the version, never the other way round. The density rule's target radius is
// CAMERA_PX_FOR / (SAMPLES(n) / n * T). That ratio used to be one constant for every format, because every format
// was 286 modules; since the rings (2026-09-27) it is MODULES / SPAN of the ring the format is painted in, the same for
// every picture in one ring, so it is taken at the format's own ring.
//
// FIVE FORMATS, not a ladder (2026-09-21; replaced by the 64-format ladder on 2026-09-23, and that by VERSIONS on
// 2026-10-01, so this is the
// history of the decision, not the rule). A rung was only ever worth having if a symbol could not simply be SCALED to
// the room instead, and LIZARD scales fractionally without penalty: scripts/exp/display_scale.mjs reads a version's
// whole payload down to about 2.2 device px a cycle at the top ring whatever the scale, and
// scripts/exp/display_bilinear.mjs says the only way to spoil it is to smooth while resampling. So the thirty-one rungs
// bought nothing a resampler does not, and they cost a format that nobody can hold in their head.
//
// Named for their sub-channel count, after AES. What a name buys is that it is the WHOLE description: the
// picture size follows from it (N_FOR), the room it needs follows from it (ROOM_FOR), and the payload is
// subch / 8 blocks. A receiver that knows which format it is looking at knows everything about the symbol.
//
// 64 is the small one: 381 device px of room against 128's 538, which is what a narrow window or a phone in
// portrait actually has.
//
// 16 is the FALLBACK, and it is a fallback rather than a rung: it needs 226 device px, a third of 256's, and
// carries 938 B a frame. A 320 CSS px phone, the smallest there is, has 640 or 960 device px at the pixel ratios
// real devices ship and picks LIZARD-192 or -256 at full screen, so this is not for the phone's screen. It is
// for the two cases that actually run short: a device at ratio 1, where 320 device px leaves even 64 tight, and
// a symbol given part of a window rather than all of it, which is the ordinary case on a page. The jump from 16
// to 64 is 4x where the others are 2x or less, and that is deliberate: 16 is somewhere to fall, not somewhere to
// sit. 8 sub-channels would be one block a frame, where a single failed block is an empty frame, so 16 is the
// floor whatever else is wanted.
// Modules of white round the frame, part of the symbol and not a request to the page: the codec's own
// (src/focus.h FOCUS_QUIET, why it is there and what it is worth), which its paint path puts round every symbol, read
// from the codec so that no copy of it lives here. Until 2026-09-23 sim/phy.mjs passed it to the painter as SAMPLES,
// so the rig painted half a module while the harnesses that converted it painted two; until 2026-09-24 the number
// lived here and the codec painted whatever it was handed. Undefined until sim/ob.mjs init() has loaded the codec,
// which PAINTED_MARGIN, and so ROOM_FOR and pickVersion, refuse.
import { QUIET } from "./ob.mjs";
export { QUIET as OB_QUIET } from "./ob.mjs";
// Modules the sender paints outside the frame, which the room it needs has to hold as well. (An optional guard ring
// round more white was a sender setting from 2026-09-23 until it was deleted as useless on 2026-09-26.)
export const PAINTED_MARGIN = () => {
  if (QUIET === undefined) throw new Error("the codec's margin is not known yet: await sim/ob.mjs init() first (or this ob.wasm predates src/focus.h FOCUS_QUIET)");
  return QUIET;
};
// Until 2026-10-01 every multiple of 16 from 16 to 1024: 64 formats (2026-09-23: even QR, whose ECC must be harsher for
// paper, has 40 versions, and a computer-generated symbol can allow more). 16 is for televisions and far away, 1024 for
// 4K capture, top-end phones and GPU decoding. The word carries the version, subch / 8, so these were versions 2, 4,
// ... 128, and a version off that ladder (LIZARD-8, -24, every odd version) was the sender's to refuse (2026-09-24): a
// receiver reads any version exactly as it reads the ladder, with no flag. Past LIZARD-1024 the codec refuses to build
// a symbol at all (src/focus.c init), since the word cannot name its version.
// This replaced LIZARD-16 / 64 / 128 / 192 / 256 (2026-09-21), whose reasoning is above and in STATUS.md.
// ai: Every whole number of blocks from 1 to 128 since 2026-10-01 (the sender's slider takes the smallest step that
// ai: works): a block is 8 sub-channels (src/focus.h FOCUS_GROUP), the word names subch / 8, so 8 is the smallest
// ai: step a symbol can take; the 64 of 16 to 1024 by 16 before.
export const VERSIONS = Array.from({ length: 128 }, (_, i) => 8 * (i + 1));
export const NAME = (subch) => `LIZARD-${subch}`;
export const versionOf = (subch) => subch / 8;   // what the format word carries; the blocks a frame are blocksFor's
// ai: The format's rate profile (2026-10-07, four rates since 2026-10-08; src/focus.c focus_tiers_for, the same
// ai: integers and order of search): [[rate, blocks, subs], ...] inner first, 7/8 blocks of 7 sub-channels on about
// ai: TIER_IN percent of them, 3/4 blocks of 8 filling, 2/3 blocks of 9 on about TIER_23 percent, 1/2 blocks of 12 on
// ai: about the outer TIER_OUT percent; the tiers with blocks only; [] for a count that is no format's. blocksFor:
// ai: the blocks a frame carries, each 473 B.
export const TIER_IN = 39, TIER_23 = 15, TIER_OUT = 20;
export function tiersFor(subch) {
  if (subch < 8 || subch % 8 || subch > 1024) return [];
  let best = -1, ba = 0, bt = 0, bc = 0;
  for (let a = 0; 7 * a <= subch - 8; a++) for (let t = 0; 7 * a + 9 * t <= subch - 8; t++) for (let c = 0; 7 * a + 9 * t + 12 * c <= subch - 8; c++) {
    if ((subch - 7 * a - 9 * t - 12 * c) % 8) continue;
    const d7 = 700 * a - TIER_IN * subch, d3 = 900 * t - TIER_23 * subch, d2 = 1200 * c - TIER_OUT * subch, cost = d7 * d7 + d3 * d3 + d2 * d2;
    if (best < 0 || cost < best) { best = cost; ba = a; bt = t; bc = c; }
  }
  return [...(ba ? [[6, ba, 7]] : []), [4, (subch - 7 * ba - 9 * bt - 12 * bc) / 8, 8], ...(bt ? [[3, bt, 9]] : []), ...(bc ? [[2, bc, 12]] : [])];
}
export const blocksFor = (subch) => tiersFor(subch).reduce((n, [, b]) => n + b, 0);
// The border in modules a side, which src/focus.c sets as the corner mark's own size plus the guard interval,
// because the border grows OUTWARDS to hold its marks and nothing ever reaches into the picture.
export const OB_MARGIN = 12 + 3;
// ai: THE RINGS (src/focus.h FOCUS_RING; three sizes on 2026-09-27, that evening 32, 64, 128, then 32, 64, 96, 128).
// ai: The dotted band holds RINGS[r] cells of 2 x 2 modules a side, so a side is 2 B + 30 modules (94 / 158 / 222 /
// ai: 286) and the picture spans 2 B (64 / 128 / 192 / 256). The ring only locates the symbol and syncs its grid, like
// ai: 5G's sync block; the format word says what is inside, and any ring may carry any picture. RING_DEFAULT is the
// ai: sender's ring for every picture unless one is named (src/focus.h FOCUS_RING_DEFAULT): the 128 since 2026-10-01
// ai: (the 64 from the evening of 2026-09-27). No ring follows the version, even by default. Until that evening the
// ai: default followed n (32 / 48 / 64), and before the rings the side itself, n / 8 + 60 (92 / 124 / 188 / 316). Keep
// ai: equal to the C.
// The ring and the picture are separate grids: the ring is painted at CELL_OF whole pixels a module and the
// picture's n samples are resampled to fill its span, so a module is n / span picture samples.
export const RINGS = [32, 64, 96, 128];
export const RING_DEFAULT = 3;
// ai: The ring helpers take (n, ring) alike; SPAN and MODULES do not depend on n, CELL_OF and SAMPLES do.
// ai: A picture's span, modules, in ring `ring` (the default unless named).
export const SPAN = (n, ring = RING_DEFAULT) => 2 * RINGS[ring];
// Modules along a side, which is what a receiver finds before it knows anything else: it identifies the ring.
export const MODULES = (n, ring = RING_DEFAULT) => SPAN(n, ring) + 2 * OB_MARGIN;
// Whole pixels a module in the painted symbol (src/focus.c pxm): the fewest that do not shrink the picture.
export const CELL_OF = (n, ring = RING_DEFAULT) => Math.ceil(n / SPAN(n, ring) - 1e-4);
// Pixels across the painted symbol, margin not included.
export const SAMPLES = (n, ring = RING_DEFAULT) => MODULES(n, ring) * CELL_OF(n, ring);
// How a block's bits lie on its coefficients, src/focus.h FOCUS_BITMAP: the sender writes it into its config so a
// recording says how it was painted (sim/phy.mjs recordedSpec). Keep the two equal.
export const FOCUS_BITMAP = 3;   // LINEAR
// Sub-channels a picture of n samples can carry. The coefficients are taken in order of increasing RADIUS, so
// subch * 320 of them fill a half-disc of radius R(subch), and that disc has to fit inside the half-plane:
// R <= n / 2, so subch <= pi (n/2)^2 / (2 * 320). Counting the half-plane's coefficients instead, which is what
// this used to do, lets the disc run out past Nyquist into the corners of the spectrum: it allowed 101
// sub-channels at n = 256 where the disc only fits 80, and the difference is measurable, not theoretical.
// At 96 sub-channels (the old cap allowed it) n = 256 delivers 2931 B a frame at 480 px where n = 512 delivers
// 4807, because the top ring's period is 1.83 samples, under the two a sample can represent.
export const R_RING = (subch) => Math.sqrt(2 * 320 * subch / Math.PI);   // the radius subch * 320 coefficients reach
export const SUBCH_CAP = (n) => Math.floor(Math.PI * n * n / (8 * 320));
// n = 128 is NOT used, and the reason in the code before this was the wrong one. It said n = 128 "cannot carry
// even version 1", true only while the ladder stepped by 32; at steps of 8 its half-plane holds 25 sub-channels,
// enough for versions 2 and 3, and it would be worth having, since a low version does not need a big picture and
// a smaller one paints at twice the device pixels a sample for a quarter of the transform.
//
// What stopped it was discovery: when every format was one module count, a receiver set up for 268 modules read 0
// of 30 n = 128 captures. The module count came to be read off the symbol, so that went, and what stopped it before
// the rings (2026-09-27) was size: n = 128 at CELL 4 was 62 modules, whose band held 17 cells a side where the format
// word needs 16 of its own beside the track's, so it had no word.
// The picture size, DERIVED and not chosen. n decides only how finely the same half-disc of coefficients is
// sampled: the top ring's period is n / R samples, so the whole of what n means is the oversampling ratio
// n / (2R). scripts/exp/focus_nfit.mjs sweeps n against the version with the symbol held at one physical size, and the
// answer is the same in every row: payload is flat once n >= 3R and falls off below it.
//
//   version  8 (R 114) at n = 256, ratio 1.12:  3465 B against 3752    -8%
//   version 24 (R 198) at n = 512, ratio 1.29:  7449 B against 7621    -2.3%   (60 frames)
//   version 32 (R 228) at n = 512, ratio 1.12:  8348 B against 8591    -2.9%   (60 frames)
//   versions 12 and 16 at ratio 1.8 and 1.6:    already the best there is; n = 1024 is 2 to 4% WORSE
//
// ai: So n >= 3R, and n is the first of PICTURE_SIZES that holds it: 2^k and 3 x 2^k from 256 to 1536 (2026-09-27;
// ai: STATUS "Picture sizes between the powers of two"). No step is more than 1.5 times the last, so n / 2R stays
// ai: between 1.5 and 2.24, where the powers of two alone put LIZARD-144 at 1024 (2.99) and LIZARD-576 at 2048: four
// ai: times the samples for one more block. scripts/exp/focus_nfit.mjs on the rings (simulator, told): 0.85% fewer
// ai: blocks over eight formats at the smaller sizes, the sampler and transform 0.62 to 0.69 of their time; the "1024
// ai: is 2 to 4% worse" above did not repeat. 1536 holds LIZARD-1024 (3R = 1370), so 2048 is no ladder size. The GPU
// ai: decoder's picture slots are this list's indices (gpu/tables.mjs PICTURES, gpu/wgsl/back_transform.mjs SIZES).
// ai: src/focus.c FOCUS_PICTURES is the same list.
export const PICTURE_SIZES = [256, 384, 512, 768, 1024, 1536];
export const N_FOR = (subch) => PICTURE_SIZES.find((n) => n >= 3 * R_RING(subch)) ?? PICTURE_SIZES.at(-1);
export const MIN_N = N_FOR;   // the old name, kept so nothing that only wants "which picture" has to change
// Camera px a cycle the top ring must still get, fitted by scripts/exp/focus_sweep.mjs (SUBCH set), 2026-09-20. It was
// 2.5, and two things were wrong with that: the border was counted in the application and not in the fit, and the
// fit averaged in cells that are not resolution-limited (blur, angle, phone, lens distortion) and cells whose
// optimum lies past the top of the sweep. On the five cells that really are resolution-limited it is 2.12, and
// the rule then predicts their measured best to within a rung: 480 exact, 720 216 against 224, fill 0.3 88
// against 96, fill 0.4 160 against 192 (1.4% of payload), and the big cells past the ladder's 256 ceiling.
export const T_PX_PER_CYCLE = 2.12;
// Camera pixels across the symbol a format NEEDS, which is a fact about the format and not a guess about anyone's
// camera: its top ring has to get T_PX_PER_CYCLE of them. T is a property of the picture, so the border's share
// ai: multiplies it at the format's own n and ring: in the 64 ring LIZARD-16 asks 149, 64 asks 299, 128 asks 423, 192
// ai: asks 518, 256 asks 598.
// Nothing chooses a format from this; it is here so a caller can ask what a format demands, and so the fitted T
// stays attached to the thing it was fitted for.
const SYMBOL_OVER_PICTURE = (subch, ring = RING_DEFAULT) => MODULES(N_FOR(subch), ring) / SPAN(N_FOR(subch), ring);   // the symbol's width over the picture's
export const CAMERA_PX_FOR = (subch, ring) => SYMBOL_OVER_PICTURE(subch, ring) * T_PX_PER_CYCLE * R_RING(subch);
// Device px a CYCLE the display must keep at the top ring, which is the same quantity the camera side is fitted
// on and for the same reason: both are sampling one band-limited picture, and what has to survive is the finest
// ring in it, not the sample grid under it. Measured in scripts/exp/display_scale.mjs, the floor is where the payload
// stops being whole:
//   version 2  (ring 8.97 samples)  0.3 px a sample = 2.7 px a cycle
//   version 12 (ring 3.66 samples)  0.6 px a sample = 2.2 px a cycle
//   version 32 (ring 2.24 samples)  0.9 px a sample = 2.0 px a cycle
// 2.7 is the highest of those, so it clears all three.
//
// This is why the picture size is NOT a scaling axis. Device px across the symbol is
// (SAMPLES(n) / n) * T_DISPLAY_CYCLE * R(subch), and n is N_FOR(subch), so the room a version needs depends on
// the SUB-CHANNELS alone. At that room every format's border module landed at 2.1 to 3.4 device px while the border
// scaled with n (until the rings, 2026-09-27), where the 286-module LIZARD-16 put it at 0.6; in one ring for every
// picture a small format's modules are small again (LIZARD-16's room in the 64 ring is 1.2 px a module). The rule
// used to ask for 1.5 device px a SAMPLE, which made a big picture expensive for no reason: version 2 asked for 429 px
// of room where it reads perfectly in 168.
//
// A page must still not upscale with SMOOTHING on: at any fractional scale that modulates the picture rather
// than blurring it, and it takes 38% at scale 2.5 (scripts/exp/display_bilinear.mjs).
export const T_DISPLAY_CYCLE = 2.7;
// Device px across the whole symbol that a version needs, the margin the sender paints round it included. No free n
// in it: n follows from the version.
// ai: ring: the ring it is painted in, its default unless named.
export const ROOM_FOR = (subch, ring = RING_DEFAULT) =>
  ((MODULES(N_FOR(subch), ring) + 2 * PAINTED_MARGIN()) / SPAN(N_FOR(subch), ring)) * T_DISPLAY_CYCLE * R_RING(subch);

// There is no camera assumption anywhere in here, and that is deliberate.
//
// It used to be scaled by the symbol's share of the screen's short side, on the reasoning that a viewer stands
// where they stand, so a symbol in a half-width window subtends half the angle. That is right for a code someone
// GLANCES at, a poster or a label. LIZARD is for optical TRANSFER: the receiver is pointed at the symbol on
// purpose and framed on it, so the operator moves or zooms until it fills the viewfinder and the pixels across it
// barely depend on its size on screen. Scaling by the share also made the top format unreachable, since
// LIZARD-256 would have needed the symbol to take 91% of the screen's short side and no window with any chrome
// does, so a normal canvas fell all the way to the fallback.
//
// With that gone the guess never bound anything, so it went too. The room is the only input.
// ai: pickVersion(roomPx, top, ring): the largest sub-channel count up to `top` whose room (ROOM_FOR) fits in roomPx
// ai: device px; the library's default top is the whole ladder. ring: the ring to paint in, RING_DEFAULT when left
// ai: out. Returns the ring too.
export function pickVersion(roomPx, top = VERSIONS.at(-1), ring = null) {
  // The ROOM decides, and nothing else does. There is no camera assumption left: the sender cannot see the
  // capture, guessing at it put a control on the page that nobody could answer, and once the guess stopped being
  // scaled by the symbol's share of the screen it never bound anything anyway. What a format needs of a camera is
  // still known and still measured (CAMERA_PX_FOR), it is just not an input to this.
  // The largest format that fits, and not the first one that does not: the room a format needs is not monotonic
  // across a picture-size step. At the step the border's share of the picture halves, so LIZARD-576 on n = 2048
  // needs 979 px where LIZARD-560 on 1024 needs 1019, and walking the ladder until something failed to fit would
  // stop at 560 in every room from 979 to 1019 and never reach the formats above it.
  // ai: That was the n / 8 + 60 border, and the rings while the default ring followed n (48 to 64 at LIZARD-144). With
  // ai: one ring for every picture the room rises with the format, so every format is reachable.
  // This is the library, so it picks over the whole ladder, and since 2026-09-29 so does the reference sender (once
  // ai: every receiver read the whole ladder; its auto stopped at LIZARD-560, APP_TOP, from 2026-09-26). The advice of
  // ai: 2026-09-23 that an app stop at LIZARD-560 came from this: the largest
  // ai: picture (576 and up; 1536 since 2026-09-27, 2048 before) registers nothing in simulation at 594 camera px
  // with 1.5 px of defocus, where LIZARD-560 still reads 3752 B a frame (STATUS.md, 2026-09-23). `top` caps the pick
  // ai: for an app that wants that.
  let subch = VERSIONS[0];
  const r = ring ?? RING_DEFAULT;
  for (const v of VERSIONS) if (v <= top && ROOM_FOR(v, r) <= roomPx && v > subch) subch = v;
  const n = N_FOR(subch), samples = SAMPLES(n, r);
  // ai: `tight` is a room too small even for the fallback (LIZARD-8, 123 px in the default ring, which no real device is). `squeezed` is a room
  // that cannot hold the picture's samples at all, which only matters to a page that paints nearest: an
  // oversampled picture survives being shown smaller than its sample count, and LIZARD-16 reads whole at 0.3
  // device px a sample. Both reported, never silent.
  return { n, subch, ring: r, span: SPAN(n, r), version: versionOf(subch), samples, scale: roomPx / samples, needPx: ROOM_FOR(subch, r),
    cameraPx: CAMERA_PX_FOR(subch, r),
    whole: Math.max(1, Math.floor(roomPx / samples)), tight: ROOM_FOR(subch, r) > roomPx, squeezed: samples > roomPx };
}
