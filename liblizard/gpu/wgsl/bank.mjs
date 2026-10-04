// ai: F3, the capped peak list, and the contract every F2 proposer (gpu/bank_<name>.mjs) writes to: for each pyramid
// ai: level, each 16 x 16 tile's KEEP strongest peaks as (x, y, sigma, response) in level-0 pixels into fixed slots of
// ai: the raw buffer (frame f's slots at f * slots, level l's at bases[l]), zero where a slot is empty, and the peaks
// ai: it found added to counts[f * 16 + 1]. A proposer module exports build(fh), ensure(fh, dimsAt, bases, slots) and
// ai: dispatch(pass, fh, B). The hand-written DoG banks that set this contract (global, tiled, perscale) were deleted
// ai: on 2026-09-26; every proposer is a trained net.
import { DIMS } from "./common.mjs";

export const KEEP = 4;                // peaks a 16 x 16 tile passes on

// ai: F3: keep the strongest cap peaks of a frame, from its tiles' slots over every level together. One workgroup a
// ai: frame: a histogram of log response (8 bins an octave above TAU), the bin where the count from the top passes the
// ai: cap, then everything above it and as much of that bin as fits. Which of the cut bin's peaks fit, and the list's
// ai: order (RANK breaks ties by list index; GATHER, by a data key since 2026-09-27, does not), follow the order the
// ai: lanes' appends land in: on the 4090 the same list, byte for byte, on
// 1,469 recorded frames over repeats, batch sizes 5 and 8 and a lane grown to 2160 (2026-09-24). A compaction in slot
// order made it fixed but cost the iGPU 0.05 ms a frame at 1080 (this stage 0.02 to 0.07 ms), so it is not taken.
export const SELECT = DIMS + /* wgsl */ `
struct Sel { slots: u32, cap: u32, tau: f32, p0: u32 }   // ai: slots: a frame's over every level
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> raw: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> peaks: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> P: Sel;
var<workgroup> hist: array<atomic<u32>, 64>;
var<workgroup> cut: u32;
var<workgroup> used: atomic<u32>;
fn bin(v: f32) -> u32 { return u32(clamp(floor(8.0 * log2(v / P.tau)), 0.0, 63.0)); }
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  let base = f * P.slots;
  for (var i = li; i < P.slots; i += 256u) {
    let v = raw[base + i].w;
    if (v > 0.0) { atomicAdd(&hist[bin(v)], 1u); }
  }
  workgroupBarrier();
  if (li == 0u) {
    var total = 0u;
    var b = 63;
    loop {
      total += atomicLoad(&hist[b]);
      if (total >= P.cap || b == 0) { break; }
      b--;
    }
    cut = u32(b);
  }
  let t = workgroupUniformLoad(&cut);
  // ai: Everything above the cut bin first (fewer than the cap by construction), then the cut bin while the cap has
  // ai: room: in one pass the two would race, and a weak peak could take a strong one's place.
  for (var round = 0u; round < 2u; round++) {
    for (var i = li; i < P.slots; i += 256u) {
      let e = raw[base + i];
      if (e.w > 0.0 && select(bin(e.w) > t, bin(e.w) == t, round == 1u)) {
        let at = atomicAdd(&used, 1u);
        if (at < P.cap) { peaks[f * P.cap + at] = e; }
      }
    }
    workgroupBarrier();
  }
  if (li == 0u) { atomicStore(&counts[f * 16u + 2u], min(atomicLoad(&used), P.cap)); }
}
`;
