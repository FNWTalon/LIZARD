// ai: F7's constants, its round and the host's ring order (glsl/register.mjs NODES, REFIT, REFIT_POST; the anchors'
// ai: refit, glsl/anchors.mjs), which the host and ../../gen/gen.mjs read too.

export const WINDOW = 12;   // modules a side of a node's template
export const SUB = 2;       // template samples a module, each way
// ai: Room for 2 nx + 2 ny - 4 nodes; the rings take 36, 68 and 132 (10, 18 and 34 a side at 158, 286 and 542
// ai: modules).
export const NODES_MAX = 136;
// ai: Room for every ring's nodes in a round's light counts (Round.light, a node at its ring's base index).
export const LIGHT_MAX = 4 * NODES_MAX;
// ai: resid's record a frame: the nodes' residuals, then LENS_SLOTS vec4f of the lens about the camera's axis that
// ai: REFIT fitted to them (COONS's Lens; 2026-10-17), which every sampler adds back.
export const LENS_SLOTS = 5, RESID = NODES_MAX + LENS_SLOTS;

// ai: F7's round ([steps, stride, step], the host's Round): +-3 modules at half a module, one template sample a module
export const ROUND = [6, SUB * SUB, 0.5];
// ai: a round's lattice side: where its samples (one a module at stride SUB^2, every one at stride 1) are a whole
// ai: number of offset steps apart, the window's WINDOW / spacing - 1 spacings a side in steps and the offsets' reach
// ai: each way (F7's round 35); else 0, each position measured directly
export const latticeSide = ([steps, stride, step]) => {
  const spacing = stride === SUB * SUB ? 1 : stride === 1 ? 1 / SUB : 0, ratio = spacing / step;
  return spacing > 0 && Math.abs(ratio - Math.round(ratio)) < 1e-6 ? (WINDOW / spacing - 1) * Math.round(ratio) + 2 * steps + 1 : 0;
};

// ai: ITERS by a search over 2 to 5 (2026-10-31; the objective blocks per GPU ms: the published lenses' blocks from
// ai: the 4090, the S26's frame): 2 read 0.012% fewer blocks than 5 for 0.9 to 1.4% of the S26's frame (REFIT 0.031 to
// ai: 0.014 ms at two a launch), 3 0.002% more for 0.6 to 0.9%.
export const WMIN = 0.3, DELTA = 0.4, ITERS = 2;

// ai: A side's residual curve takes 1 to SIDE_MAX coefficients (a constant to a quintic in the position along it),
// ai: the count chosen per side and frame by AICc on the side's own nodes (2026-09-29: each frame decided from its own
// ai: evidence, with no constant set on a phone). SIDE_MAX is the power basis's conditioning on
// ai: [-1, 1] at the rings' node counts, not a fitted value.
export const SIDE_MAX = 6;

export const SIDE_NODES = NODES_MAX / 4 + 1;

export const LENS_P = 11;

export const lensIdx = (a, b) => a * LENS_P - (a * (a - 1)) / 2 + (b - a);   // ai: packed upper triangle, 66 entries

// Host side of the same order: the ring's lattice coordinates.
export function ringOrder(nx, ny) {
  const out = [];
  for (let i = 0; i < nx; i++) out.push([i, 0]);
  for (let j = 1; j < ny; j++) out.push([nx - 1, j]);
  for (let i = nx - 2; i >= 0; i--) out.push([i, ny - 1]);
  for (let j = ny - 2; j >= 1; j--) out.push([0, j]);
  return out;
}
