// F9: the picture, n x n samples a frame, through the frame's map from module coordinates to image pixels.
// Sample (x, y) sits at module (g0 + x step, g0 + y step) (src/focus.c focus_finish_bits). Written in the strip
// order the reference back half reads (src/focus.c:872): element (x, y) at (x / 64) n 64 + y 64 + x % 64.
// ai: g0 and step are the frame's (ring, picture) pair's since 2026-09-27: any ring may carry any picture.
//
// The map is common.mjs MAP (a homography and radial distortion) plus what F7 measured it to miss at the border
// nodes, spread over the picture as a Coons patch: the four sides' residual curves blended across, less the
// corners counted twice. Residuals are zero when the host hands the map over, so the patch then adds nothing.
//
// With the GPU back half and no grids asked for, this stage does not run: the back half's pass 1 samples each
// column itself through COONS below and the same map and bilinear read (wgsl/back_transform.mjs, fused).
import { DIMS, LUM0, MAP, SLOTS } from "./common.mjs";
import { RING, NODES_MAX } from "./register.mjs";

// The Coons patch in pieces, shared with the fused pass 1: a side's curve depends on one coordinate only, so that
// pass takes the top and bottom once a column and the left and right once a row, through this same arithmetic.
// ai: A shader including it binds `resid` and defines latAt(i), lattice coordinate i (a ring's nodes from `first`).
export const COONS = RING + /* wgsl */ `
fn res(f: u32, n: u32) -> vec2f { return resid[f * ${NODES_MAX}u + n].xy; }
// The residual curve along one side at coordinate t (0 top, 1 right, 2 bottom, 3 left).
fn curve(f: u32, side: u32, t: f32, first: u32, nn: u32) -> vec2f {
  let x0 = latAt(first);
  let k = min(u32(max(floor((t - x0) / 16.0), 0.0)), nn - 2u);
  let a = latAt(first + k);
  let b = latAt(first + k + 1u);
  let w = clamp((t - a) / (b - a), 0.0, 1.0);
  var i0 = k; var j0 = 0u; var i1 = k + 1u; var j1 = 0u;
  if (side == 1u) { i0 = nn - 1u; j0 = k; i1 = nn - 1u; j1 = k + 1u; }
  if (side == 2u) { j0 = nn - 1u; j1 = nn - 1u; }
  if (side == 3u) { i0 = 0u; j0 = k; i1 = 0u; j1 = k + 1u; }
  return mix(res(f, ring(i0, j0, nn, nn)), res(f, ring(i1, j1, nn, nn)), w);
}
// The blend weight across the patch: where t sits between the first node and the last, 0..1.
fn coonsT(t: f32, first: u32, nn: u32) -> f32 {
  let x0 = latAt(first);
  let x1 = latAt(first + nn - 1u);
  return clamp((t - x0) / (x1 - x0), 0.0, 1.0);
}
struct Corners { c00: vec2f, c10: vec2f, c01: vec2f, c11: vec2f }
fn corners(f: u32, nn: u32) -> Corners {
  return Corners(res(f, ring(0u, 0u, nn, nn)), res(f, ring(nn - 1u, 0u, nn, nn)), res(f, ring(0u, nn - 1u, nn, nn)), res(f, ring(nn - 1u, nn - 1u, nn, nn)));
}
fn coonsMix(u: f32, v: f32, top: vec2f, bot: vec2f, lft: vec2f, rgt: vec2f, c: Corners) -> vec2f {
  return (1.0 - v) * top + v * bot + (1.0 - u) * lft + u * rgt
       - ((1.0 - u) * (1.0 - v) * c.c00 + u * (1.0 - v) * c.c10 + (1.0 - u) * v * c.c01 + u * v * c.c11);
}
fn coons(f: u32, first: u32, nn: u32, m: vec2f) -> vec2f {
  return coonsMix(coonsT(m.x, first, nn), coonsT(m.y, first, nn), curve(f, 0u, m.x, first, nn), curve(f, 2u, m.x, first, nn),
    curve(f, 3u, m.y, first, nn), curve(f, 1u, m.y, first, nn), corners(f, nn));
}
`;

// ai: sizes[SLOTS ring + slot] = (n, g0, step, 0): picture slot `slot` (gpu/tables.mjs PICTURES) in ring `ring`, the
// ai: pair's grid (gpu/tables.mjs gridOf); the frame's pair is sel's (ring, picture slot), F8's since 2026-09-27.
export const SAMPLE = DIMS + MAP + /* wgsl */ `
struct Size { n: u32, g0: f32, step: f32, p0: u32 }
struct Grid { stride: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> frames: array<Frame>;
@group(0) @binding(2) var<storage, read> maps: array<array<f32, 16>>;
@group(0) @binding(3) var<storage, read> sizes: array<Size>;
@group(0) @binding(4) var<storage, read_write> grid: array<f32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> G: Grid;
// ai: The frame's ring, and the version (y = 1 + it) and picture slot F8 gave it (wgsl/word.mjs): y under 2 samples
// ai: nothing, and neither does a picture past the grid's room (a version naming n = 1536 in a lane made for 1024).
@group(0) @binding(7) var<storage, read> sel: array<vec4u>;
// resid[f][n]: each border node's residual in image pixels, ring order (wgsl/register.mjs). lattice: the node
// ai: coordinates of every ring (the same both ways), at L.at[ring] = (first, nodes a side, 0, 0).
struct Lattice { at: array<vec4u, 4> }
@group(0) @binding(8) var<storage, read> resid: array<vec4f>;
@group(0) @binding(9) var<storage, read> lattice: array<f32>;
@group(0) @binding(10) var<uniform> LT: Lattice;
fn latAt(i: u32) -> f32 { return lattice[i]; }
` + COONS + /* wgsl */ `
var<workgroup> wrote: atomic<u32>;
` + LUM0 + /* wgsl */ `
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = g.z;
  let F = frames[f];
  let pick = sel[f];
  let S = sizes[${SLOTS}u * pick.x + pick.z];
  let live = F.valid != 0u && pick.y > 1u && S.n * S.n <= G.stride && g.x < S.n && g.y < S.n;
  if (live) {
    let m = vec2f(S.g0 + f32(g.x) * S.step, S.g0 + f32(g.y) * S.step);
    let at = LT.at[pick.x];
    let p = toImage(maps[f], m) + coons(f, at.x, at.y, m);
    grid[f * G.stride + (g.x / 64u) * S.n * 64u + g.y * 64u + (g.x % 64u)] = bilin0(f, p, i32(F.w), i32(F.h));
    atomicAdd(&wrote, 1u);
  }
  workgroupBarrier();
  if (li == 0u) {
    let k = atomicLoad(&wrote);
    if (k > 0u) { atomicAdd(&counts[f * 16u + 0u], k); }
  }
}
`;
