// The sampler's map pieces in GLSL: the WGSL sample.mjs (archived) COONS statement for statement (its comments): the border nodes'
// residual curves along each side, blended across the picture, less the corners counted twice, with RING and LENS from
// ./register.mjs (where the WGSL sample.mjs (archived) takes them from the WGSL register.mjs (archived)). Pass 1, the anchors and F8 read the
// final map through these.
// ai: A shader including COONS binds `resid` (vec4 a node, RESID a frame), defines latAt(i), lattice coordinate i,
// ai: before it, as the WGSL's do, and includes CONVERT first. F9's own kernel (SAMPLE) is built only for the wasm back
// ai: half, which the native tree does not build, so it has no GLSL twin.
import { RESID } from "../plan/register.mjs";
import { RING, LENS } from "./register.mjs";

export const COONS = RING + LENS + /* glsl */ `
vec2 res(uint f, uint n) { return resid[f * ${RESID}u + n].xy; }
vec2 curve(uint f, uint side, float t, uint first, uint nn) {
  float x0 = latAt(first);
  uint k = min(f2u(max(floor((t - x0) / 16.0), 0.0)), nn - 2u);
  float a = latAt(first + k);
  float b = latAt(first + k + 1u);
  float w = clamp((t - a) / (b - a), 0.0, 1.0);
  uint i0 = k; uint j0 = 0u; uint i1 = k + 1u; uint j1 = 0u;
  if (side == 1u) { i0 = nn - 1u; j0 = k; i1 = nn - 1u; j1 = k + 1u; }
  if (side == 2u) { j0 = nn - 1u; j1 = nn - 1u; }
  if (side == 3u) { i0 = 0u; j0 = k; i1 = 0u; j1 = k + 1u; }
  return mix(res(f, ring(i0, j0, nn, nn)), res(f, ring(i1, j1, nn, nn)), w);
}
float coonsT(float t, uint first, uint nn) {
  float x0 = latAt(first);
  float x1 = latAt(first + nn - 1u);
  return clamp((t - x0) / (x1 - x0), 0.0, 1.0);
}
struct Corners { vec2 c00; vec2 c10; vec2 c01; vec2 c11; };
Corners corners(uint f, uint nn) {
  return Corners(res(f, ring(0u, 0u, nn, nn)), res(f, ring(nn - 1u, 0u, nn, nn)), res(f, ring(0u, nn - 1u, nn, nn)), res(f, ring(nn - 1u, nn - 1u, nn, nn)));
}
vec2 coonsMix(float u, float v, vec2 top, vec2 bot, vec2 lft, vec2 rgt, Corners c) {
  return (1.0 - v) * top + v * bot + (1.0 - u) * lft + u * rgt
       - ((1.0 - u) * (1.0 - v) * c.c00 + u * (1.0 - v) * c.c10 + (1.0 - u) * v * c.c01 + u * v * c.c11);
}
vec2 coons(uint f, uint first, uint nn, vec2 m) {
  return coonsMix(coonsT(m.x, first, nn), coonsT(m.y, first, nn), curve(f, 0u, m.x, first, nn), curve(f, 2u, m.x, first, nn),
    curve(f, 3u, m.y, first, nn), curve(f, 1u, m.y, first, nn), corners(f, nn));
}
`;
