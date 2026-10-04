// ai: The sender's GPU encoder after the inverse transform (gpu/encoder.mjs; 2026-09-29, as the sender has lagged at
// ai: LIZARD-1024; bit exact only in the ring). The picture comes from wgsl/cancel_paint.mjs in its "send" mode: PIC,
// ai: n x n f32 a symbol in 0..1, clipped. Three shaders, the C's arithmetic (src/focus.c resample, focus_paint_rgba):
// ai: TPOSE, the picture row-major: ipic leaves it quad-major (PICQ[x / 4][y], four columns' rows consecutive, so its
// ai:   stores are whole lines); a tile of 16 quads by 16 rows a workgroup through workgroup memory, read a quad's 16
// ai:   rows (256 B) at a time and written a row's 16 quads at a time, so both sides move whole lines.
// ai: RSV, down the picture: TMP[y][k] = sum over t < 6 of w[6 y + t] pic[idx[6 y + t]][k], y < q, k < n, a thread
// ai:   four adjacent k (vec4f), the taps the C's (sim/ob.mjs resampleGeom), idx its first samples wrapped mod n on the
// ai:   host: the picture read as periodic, so the q pixels cover the picture and its guard interval, which is the
// ai:   picture's own wrap.
// ai: RSH, across and out: a thread one u32 of a frame row, four bytes, the codes symbols side by side (FW = codes W
// ai:   pixels a row, W rows, each row padded to RW words: W need not be a multiple of four, pxm 1 in the 128 ring at
// ai:   n = 256): inside a symbol's square (q pixels from sq) the same six taps along the row of TMP, clamped to 0..1,
// ai:   then (drive 255 + 0.5) truncated and held to 255 as focus_paint_rgba does; anywhere else the border's byte
// ai:   (BORDER, one symbol W x W, the C's own paint: ring, marks, band, word, quiet margin), so the border is the C's
// ai:   to the byte and nothing border-shaped is drawn here. No integer division: a row a workgroup y, the symbol by
// ai:   subtraction (2026-09-29: with the frame's words flat and the wrap taken mod n a tap, 7.35 ms of the iGPU's
// ai:   13.6 at LIZARD-1024).
// ai: A copy (every pixel's taps the C's on-sample one, [0, 0, 1, 0, 0, 0], wherever the ring's span divides n: every
// ai: picture in the 32 and 64 rings, 384, 768 and 1536 in the 96, all but 384 in the 128, 20 of the 24 pairs, SPEC
// ai: 6.8): RSV is not run and RSH reads the picture at the one tap's sample on both axes, the value the two passes
// ai: would have given exactly (rshSource({ copy: true })).
// ai: PRESENT, a render pass onto the canvas: pixel (x, y) shows the frame's byte (x / whole, y / whole), grey.
// ai: The tolerance: the square within one grey level of the C's (its FFT's rounding differs), the border equal.

export const TPOSE_TILE = 16;
export const RSV_THREADS = 64;
export const RSH_THREADS = 64;

// ai: G: a = (n, q, sq, W): the picture, the square's pixels, its first pixel in a symbol (margin included), the
// ai: symbol's width; c = (whole, f0, gap, 0): the present's whole canvas px a frame px, an encode's first frame, and the gap
// ai: between two codes in px (gpu/encoder.mjs GAP_MODULES modules, 2026-09-30); b = (codes, FW, RW, FS): symbols a frame, the frame's width in pixels, a row's in u32 and a frame's
// ai: in a batch's FRAME (RW W, padded to 256 B so PRESENT binds a frame at its offset); c = (whole, f0, 0, 0): canvas
// ai: pixels a frame pixel (PRESENT) and the frame of FRAME the dispatch's first writes (RSH). TAPS: idx (6 q u32, a
// ai: tap's sample mod n), then w (6 q f32, bitcast), both passes alike. A dispatch is frames x codes symbols of
// ai: scratch, symbol r = frame codes + code, a workgroup z each (RSH: a frame, written to frame f0 + z of FRAME).
const G = /* wgsl */ `
struct G { a: vec4u, b: vec4u, c: vec4u }
`;

// ai: Bindings: PICQ (ro, vec4f), PIC (rw, vec4f), G. Dispatch (n / 4 / TPOSE_TILE, n / TPOSE_TILE, symbols), 64 threads.
export function tposeSource() {
  return G + /* wgsl */ `
@group(0) @binding(0) var<storage, read> PICQ: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> PIC: array<vec4f>;
@group(0) @binding(2) var<uniform> P: G;
var<workgroup> tile: array<vec4f, ${TPOSE_TILE * TPOSE_TILE}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) i: u32) {
  let n: u32 = P.a.x;
  let n4: u32 = n / 4u;
  let base: u32 = wg.z * n4 * n;
  let q0: u32 = wg.x * ${TPOSE_TILE}u;
  let y0: u32 = wg.y * ${TPOSE_TILE}u;
  for (var j: u32 = 0u; j < ${(TPOSE_TILE * TPOSE_TILE) / 64}u; j = j + 1u) {
    let e: u32 = i + 64u * j;
    let qd: u32 = e / ${TPOSE_TILE}u;
    let r: u32 = e % ${TPOSE_TILE}u;
    tile[r * ${TPOSE_TILE}u + qd] = PICQ[base + (q0 + qd) * n + y0 + r];
  }
  workgroupBarrier();
  for (var j: u32 = 0u; j < ${(TPOSE_TILE * TPOSE_TILE) / 64}u; j = j + 1u) {
    let e: u32 = i + 64u * j;
    let r: u32 = e / ${TPOSE_TILE}u;
    let qd: u32 = e % ${TPOSE_TILE}u;
    PIC[base + (y0 + r) * n4 + q0 + qd] = tile[e];
  }
}
`;
}

// ai: Bindings: PIC (ro, vec4f), TAPS (ro), TMP (rw, vec4f), G. Dispatch (ceil(n / 4 / RSV_THREADS), q, symbols).
export function rsvSource() {
  return G + /* wgsl */ `
@group(0) @binding(0) var<storage, read> PIC: array<vec4f>;
@group(0) @binding(1) var<storage, read> TAPS: array<u32>;
@group(0) @binding(2) var<storage, read_write> TMP: array<vec4f>;
@group(0) @binding(3) var<uniform> P: G;
@compute @workgroup_size(${RSV_THREADS})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n4: u32 = P.a.x / 4u;
  let q: u32 = P.a.y;
  let k4: u32 = id.x;
  let y: u32 = id.y;
  let r: u32 = id.z;
  if (k4 >= n4) { return; }
  let base: u32 = r * P.a.x * n4 + k4;
  var v: vec4f = bitcast<f32>(TAPS[6u * q + 6u * y]) * PIC[base + TAPS[6u * y] * n4];
  for (var t: u32 = 1u; t < 6u; t = t + 1u) {
    v = v + bitcast<f32>(TAPS[6u * q + 6u * y + t]) * PIC[base + TAPS[6u * y + t] * n4];
  }
  TMP[(r * q + y) * n4 + k4] = v;
}
`;
}

// ai: Bindings: TMP (ro, f32; with copy the picture, PIC, n x n a symbol), TAPS (ro), BORDER (ro, one symbol's bytes,
// ai: four a u32), FRAME (rw, a half's frames FS words apart), G. Dispatch (ceil(RW / RSH_THREADS), W, frames).
export function rshSource({ copy = false } = {}) {
  const inside = copy ? /* wgsl */ `
    let v: f32 = clamp(TMP[k * P.a.x * P.a.x + TAPS[6u * (y - sq) + 2u] * P.a.x + TAPS[6u * xx + 2u]], 0.0, 1.0);` : /* wgsl */ `
    let base: u32 = (k * q + (y - sq)) * P.a.x;
    var v: f32 = bitcast<f32>(TAPS[6u * q + 6u * xx]) * TMP[base + TAPS[6u * xx]];
    for (var t: u32 = 1u; t < 6u; t = t + 1u) {
      v = v + bitcast<f32>(TAPS[6u * q + 6u * xx + t]) * TMP[base + TAPS[6u * xx + t]];
    }
    v = clamp(v, 0.0, 1.0);`;
  return G + /* wgsl */ `
@group(0) @binding(0) var<storage, read> TMP: array<f32>;
@group(0) @binding(1) var<storage, read> TAPS: array<u32>;
@group(0) @binding(2) var<storage, read> BORDER: array<u32>;
@group(0) @binding(3) var<storage, read_write> FRAME: array<u32>;
@group(0) @binding(4) var<uniform> P: G;
fn pixel(x: u32, y: u32, f: u32) -> u32 {
  let q: u32 = P.a.y;
  let sq: u32 = P.a.z;
  let W: u32 = P.a.w;
  var xs: u32 = x;
  var k: u32 = f * P.b.x;
  // ai: the codes W wide with a gap of P.c.z between them (gpu/encoder.mjs GAP_MODULES): a gap pixel takes the
  // ai: margin's byte beside it
  loop { if (xs < W + P.c.z) { break; } xs = xs - W - P.c.z; k = k + 1u; }
  if (xs >= W) { xs = W - 1u; }
  if (xs >= sq && xs < sq + q && y >= sq && y < sq + q) {
    let xx: u32 = xs - sq;` + inside + /* wgsl */ `
    return min(u32(v * 255.0 + 0.5), 255u);
  }
  let j: u32 = y * W + xs;
  return (BORDER[j >> 2u] >> (8u * (j & 3u))) & 0xffu;
}
@compute @workgroup_size(${RSH_THREADS})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let wx: u32 = id.x;
  let y: u32 = id.y;
  let f: u32 = id.z;
  if (wx >= P.b.z) { return; }
  var word: u32 = 0u;
  for (var b: u32 = 0u; b < 4u; b = b + 1u) {
    let x: u32 = 4u * wx + b;
    if (x < P.b.y) { word = word | (pixel(x, y, f) << (8u * b)); }
  }
  FRAME[(P.c.y + f) * P.b.w + y * P.b.z + wx] = word;
}
`;
}

// ai: Bindings (fragment): FRAME (ro, one frame: bound at its offset in the batch's), G. A triangle over the canvas; no
// ai: vertex buffer.
export function presentSource() {
  return G + /* wgsl */ `
@group(0) @binding(0) var<storage, read> FRAME: array<u32>;
@group(0) @binding(1) var<uniform> P: G;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p: vec2f = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let whole: u32 = max(P.c.x, 1u);
  let x: u32 = min(u32(pos.x) / whole, P.b.y - 1u);
  let y: u32 = min(u32(pos.y) / whole, P.a.w - 1u);
  let i: u32 = 4u * P.b.z * y + x;
  let g: f32 = f32((FRAME[i >> 2u] >> (8u * (i & 3u))) & 0xffu) / 255.0;
  return vec4f(g, g, g, 1.0);
}
`;
}
