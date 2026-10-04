// WGSL shared by the stages. WGSL has no includes, so a stage's source is these pieces joined in JS.
import { PICTURE_SIZES } from "../../sim/lizard_pick.mjs";

// ai: The picture sizes (sim/lizard_pick.mjs PICTURE_SIZES; gpu/tables.mjs PICTURES), a frame's picture slot its
// ai: index here (sel.z, from F8's version), and SLOTS of them. Every per-slot table on the device (F8's tops, F9's
// ai: sizes, the back half's gate, LISTS, ARGS, the LDPC's and the paint's uniforms, gate2's) is SLOTS long, so a
// ai: ladder change edits no stage. Here, not in back_transform.mjs, because sample.mjs needs it and back_transform.mjs
// ai: imports sample.mjs.
export const SIZES = PICTURE_SIZES;
export const SLOTS = SIZES.length;
//
// Frame f of a batch is layer f of the level-0 texture (r8unorm, the capture as it came) and a slice of each
// ai: pyramid buffer. dims[f] = (w, h, 0, valid), the third word unread since the rings (2026-09-27: a frame's ring
// ai: and picture ride in sel): frames in one batch may differ in size, and a slot with
// valid = 0 is empty and does nothing.
export const DIMS = /* wgsl */ `
struct Frame { w: u32, h: u32, size: u32, valid: u32 }
`;

// Level 0: bilinear, pixel centres at i + 0.5, clamped to the frame's own edge (src/internal.h sample()).
export const LUM0 = /* wgsl */ `
fn lum0(f: u32, x: i32, y: i32, w: i32, h: i32) -> f32 {
  return textureLoad(img, vec2i(clamp(x, 0, w - 1), clamp(y, 0, h - 1)), i32(f), 0).r;
}
fn bilin0(f: u32, p: vec2f, w: i32, h: i32) -> f32 {
  let q = p - vec2f(0.5);
  let q0 = floor(q);
  let a = q - q0;
  let i = vec2i(q0);
  let t = mix(lum0(f, i.x, i.y, w, h), lum0(f, i.x + 1, i.y, w, h), a.x);
  let b = mix(lum0(f, i.x, i.y + 1, w, h), lum0(f, i.x + 1, i.y + 1, w, h), a.x);
  return mix(t, b, a.y);
}
`;

// A frame's map from module coordinates to image pixels, 16 floats: a homography to undistorted image
// coordinates (H, row major, [0..8]), then radial distortion about a centre, [9] k1, [10..11] the centre,
// [12] 1 / R^2, the same form as the camera model (scripts/sim/camera.mjs project): u = c + (p - c)(1 + k1 |p - c|^2 / R^2)
// with u undistorted and p the pixel. k1 = 0 is a plain homography.
export const MAP = /* wgsl */ `
fn toImage(M: array<f32, 16>, m: vec2f) -> vec2f {
  let w = M[6] * m.x + M[7] * m.y + M[8];
  let u = vec2f(M[0] * m.x + M[1] * m.y + M[2], M[3] * m.x + M[4] * m.y + M[5]) / w;
  if (M[9] == 0.0) { return u; }
  let c = vec2f(M[10], M[11]);
  var p = u;
  for (var i = 0; i < 6; i++) { let d = p - c; p = c + (u - c) / (1.0 + M[9] * dot(d, d) * M[12]); }
  return p;
}
`;
