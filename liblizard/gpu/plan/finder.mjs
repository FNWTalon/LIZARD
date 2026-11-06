// ai: F5 and F6's constants (glsl/finder.mjs VOTE, PEAKS, GATHER, COMPLETE, TRACE, SCORE, PICK, REDO), which the host and
// ai: ../../gen/gen.mjs (the native tree's consts) read too.
import { RINGS as RING_CELLS } from "../../sim/lizard_pick.mjs";

// ai: A mark's centre depth in modules by reading form: the gapped core, the merged square.
export const CENTRE = [8, 6];
export const K = 4;          // vote peaks a frame, so quads a frame
export const CELL = 16;      // accumulator cell, level-0 pixels
export const ORIENT = 8;     // 4 rotations x mirror
// ai: The rings a hypothesis may be (sim/lizard_pick.mjs RINGS, three): an accumulator and a module count each. Four at most:
// ai: the stages' uniforms hold the rings' module counts, node counts and lattices in one vec4 (P5, Pick, register.mjs
// ai: Fit and Round, sample.mjs Lattice, gpu/decoder.mjs).
export const RING_COUNT = RING_CELLS.length;
if (RING_COUNT > 4) throw new Error(`${RING_COUNT} rings: the finder's uniforms hold four`);
// ai: GATHER's reach runs from the first ring's to the last's, so the rings go smallest first.
if (RING_CELLS.some((c, i) => i && c <= RING_CELLS[i - 1])) throw new Error(`rings ${RING_CELLS.join(", ")}: smallest first`);
// ai: SCORE reads each quad twice, through the marks' homography alone and through the quad's traced bend (TRACE), and
// ai: the better read wins: near the corners, where the track is read, an ordinary lens is the homography's and a
// ai: fisheye the bend's.
export const BENDS = 2;
export const HYP = K * ORIENT * RING_COUNT * BENDS;   // ai: hypotheses a frame, the bend the top bit
// ai: A reading votes, and supports a quad, when the network's mark probability p reaches SMIN (the trainer reports
// ai: at 0.5).
export const SMIN = 0.35;
// ai: VOTE's readings a workgroup (2026-10-21; 64 before): a workgroup's voting readings walk their rays one after another
// ai: on one lane, so at a launch of a frame or two the stage took the longest such chain over the frame's 4 workgroups.
export const VOTE_READS = 16;
// A vote ray runs from RAY_LO to RAY_HI times the reach the corner's own module size implies (perspective moves
// the centre within that), in a band RAY_SPREAD of the reach wide (about 17 degrees either side of the corner's
// diagonal; VOTE's band is clamped at 6 cells, GATHER's cone is not). Wide enough that a frame's four corners gather
// in the first round where perspective, a lens's bend or the classifier's angle turn a diagonal off the centre, so
// the second round (REDO) runs on fewer frames.
export const RAY_LO = 0.45, RAY_HI = 2.2, RAY_SPREAD = 0.3;
// F6 reads the track only this many modules from a corner: the four corners' homography is exact there, while lens
// distortion bows the middle of a side by modules (k1 = 0.06). Orientation and module count need no more.
export const NEAR = 40;

export const SUPPRESS = 10;
// ai: The peaks a frame's list holds: any two lie SUPPRESS + 1 cells apart, so a 4K frame's 240 x 135 cells hold at
// ai: most 286.
export const PEAK_LIST = 512;

export const STATIONS = 32;

// ai: The second round's grids (2026-10-30): R2_N dispatches (the bank's round 2, then SELECT to PICK; decoder.mjs
// ai: grids2, core/dec/front.cpp its twin), each a vec4u in R2: its grid for one frame (x, y, z) and w the axis that
// ai: counts frames (3: unused). REDO raises that axis of args2 to this frame's end for every frame it sends round again,
// ai: and writes the other two, so a launch whose frames the first round all found dispatches none of it, and every
// ai: grid ends at the last frame that has work. The host clears args2 a batch.
export const R2_N = 12;
