// The pyramid and each frame's grey levels, one kernel (2026-09-30, the small stages merged for organization;
// before, the histogram was a dispatch of its own, wgsl/histogram.mjs, and each level another).
// Level l + 1 is the 2 x 2 mean of level l. Level 0 is the capture as it came (an r8unorm texture layer a frame);
// levels 1 to 4 are f32 buffers, frame f's slice at f * stride * rows (LD.d[l] = (stride, rows), the lane's). A frame
// keeps its own size at every level (ceil(w / 2^l)), and reads past its edge clamp to it.
// ai: A workgroup takes a 64 x 64 block of a frame's layer: level 1's 32 x 32 of it, four pixels a lane with adjacent
// ai: lanes on adjacent pixels (so the layer's reads and the level's writes are whole lines), then level 2 (16 x 16, a
// ai: lane each), 3 (8 x 8) and 4 (4 x 4) from the level before in workgroup memory: a level's 2 x 2 never crosses a
// ai: block's edge, and a read clamped at the frame's edge lands on the pixel beside it, in the same block. Every mean
// ai: is summed as the separate passes summed it (from 0, the upper row then the lower, each left then right, then a
// ai: quarter), so the levels are theirs bit for bit (scripts/pages/gpu_selftest_stages.mjs hashes them).
// ai: The grey levels, for the receiver's stats row (greyRange, greyMedian, greyClipHi, greyClipLo): a HIST_BINS-bin
// ai: histogram of the middle half of the frame's crop, every second pixel each way, the pixels lizard-web/recv-worker.mjs
// ai: and lizard-web/recv-gpu-worker.mjs levelsOf count on a luma frame, so a frame that came as a VideoFrame (F0: its luma
// ai: never reaches the CPU) reports the same fields. Of the four pixels a level-1 pixel reads, the one on the
// ai: window's parity is counted in workgroup bins, then the bins a workgroup filled are added to the frame's. It writes
// ai: only its own buffer: no decode stage reads it. Dispatched (ceil(W / 64), ceil(H / 64), frames); the histogram is
// ai: cleared before each batch.
import { DIMS } from "./common.mjs";

export const HIST_BINS = 256;
export const PYRAMID_BLOCK = 64;   // ai: layer pixels a workgroup's block is a side

// ai: The mean of the 2 x 2 under pixel (x, y) of level l + 1 from tile t (side s) of level l, the block's own pixel
// ai: (lx, ly) of it: the right column and the lower row clamp onto the pixel beside them past the frame's edge (w x
// ai: h, level l's), as min(2x + 1, w - 1) did.
const down = (t, s, w, h) => /* wgsl */ `
    let c0 = 2u * lx; let c1 = select(2u * lx + 1u, 2u * lx, 2u * x + 1u >= ${w});
    let r0 = 2u * ly * ${s}u; let r1 = select((2u * ly + 1u) * ${s}u, 2u * ly * ${s}u, 2u * y + 1u >= ${h});
    let v = mean4(${t}[r0 + c0], ${t}[r0 + c1], ${t}[r1 + c0], ${t}[r1 + c1]);`;

export const PYRAMID = DIMS + /* wgsl */ `
struct LevelDims { d: array<vec4u, 5> }
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> frames: array<Frame>;
@group(0) @binding(2) var<storage, read_write> lv1: array<f32>;
@group(0) @binding(3) var<storage, read_write> lv2: array<f32>;
@group(0) @binding(4) var<storage, read_write> lv3: array<f32>;
@group(0) @binding(5) var<storage, read_write> lv4: array<f32>;
@group(0) @binding(6) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> LD: LevelDims;
var<workgroup> bins: array<atomic<u32>, ${HIST_BINS}>;
var<workgroup> t1: array<f32, 1024>;   // ai: the block's level 1, 32 x 32
var<workgroup> t2: array<f32, 256>;    // ai: level 2, 16 x 16
var<workgroup> t3: array<f32, 64>;     // ai: level 3, 8 x 8
fn dimAt(v: u32, level: u32) -> u32 { return (v + (1u << level) - 1u) >> level; }
fn mean4(tl: f32, tr: f32, bl: f32, br: f32) -> f32 {
  var s = 0.0;
  s += tl;
  s += tr;
  s += bl;
  s += br;
  return 0.25 * s;
}
fn tex(f: u32, x: u32, y: u32, w: u32, h: u32) -> f32 {
  return textureLoad(img, vec2i(i32(min(x, w - 1u)), i32(min(y, h - 1u))), i32(f), 0).r;
}
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  atomicStore(&bins[li], 0u);
  workgroupBarrier();
  let f = wg.z;
  let F = frames[f];
  let live = F.valid != 0u;
  let w1 = dimAt(F.w, 1u); let h1 = dimAt(F.h, 1u);
  let w2 = dimAt(F.w, 2u); let h2 = dimAt(F.h, 2u);
  let w3 = dimAt(F.w, 3u); let h3 = dimAt(F.h, 3u);
  let w4 = dimAt(F.w, 4u); let h4 = dimAt(F.h, 4u);
  // ai: the histogram's window: the middle half, [w / 4, 3 w / 4) x [h / 4, 3 h / 4)
  let win = vec4u(F.w >> 2u, F.h >> 2u, (3u * F.w) >> 2u, (3u * F.h) >> 2u);
  // ai: Level 1: pixel (lx, ly) of the block's 32 x 32 at i = li + 256 k, a row of 32 every 32 lanes.
  for (var k = 0u; k < 4u; k++) {
    let i = li + 256u * k;
    let lx = i & 31u;
    let ly = i >> 5u;
    let x = 32u * wg.x + lx;
    let y = 32u * wg.y + ly;
    var v = 0.0;
    if (live) {
      let p00 = tex(f, 2u * x, 2u * y, F.w, F.h);
      let p01 = tex(f, 2u * x + 1u, 2u * y, F.w, F.h);
      let p10 = tex(f, 2u * x, 2u * y + 1u, F.w, F.h);
      let p11 = tex(f, 2u * x + 1u, 2u * y + 1u, F.w, F.h);
      v = mean4(p00, p01, p10, p11);
      if (x < w1 && y < h1) { lv1[(f * LD.d[1].y + y) * LD.d[1].x + x] = v; }
      // ai: the one of the four on the window's parity each way, counted where it lies in the window (it is then in
      // ai: the frame, so its read was not clamped)
      let ox = ((2u * x) ^ win.x) & 1u;
      let oy = ((2u * y) ^ win.y) & 1u;
      let sx = 2u * x + ox;
      let sy = 2u * y + oy;
      if (sx >= win.x && sx < win.z && sy >= win.y && sy < win.w) {
        let s = select(select(p00, p01, ox == 1u), select(p10, p11, ox == 1u), oy == 1u);
        atomicAdd(&bins[u32(round(clamp(s, 0.0, 1.0) * 255.0))], 1u);
      }
    }
    t1[i] = v;
  }
  workgroupBarrier();
  {
    let lx = li & 15u;
    let ly = li >> 4u;
    let x = 16u * wg.x + lx;
    let y = 16u * wg.y + ly;${down("t1", 32, "w1", "h1")}
    if (live && x < w2 && y < h2) { lv2[(f * LD.d[2].y + y) * LD.d[2].x + x] = v; }
    t2[li] = v;
  }
  workgroupBarrier();
  if (li < 64u) {
    let lx = li & 7u;
    let ly = li >> 3u;
    let x = 8u * wg.x + lx;
    let y = 8u * wg.y + ly;${down("t2", 16, "w2", "h2")}
    if (live && x < w3 && y < h3) { lv3[(f * LD.d[3].y + y) * LD.d[3].x + x] = v; }
    t3[li] = v;
  }
  workgroupBarrier();
  if (li < 16u) {
    let lx = li & 3u;
    let ly = li >> 2u;
    let x = 4u * wg.x + lx;
    let y = 4u * wg.y + ly;${down("t3", 8, "w3", "h3")}
    if (live && x < w4 && y < h4) { lv4[(f * LD.d[4].y + y) * LD.d[4].x + x] = v; }
  }
  let n = atomicLoad(&bins[li]);
  if (n != 0u) { atomicAdd(&hist[f * ${HIST_BINS}u + li], n); }
}
`;
