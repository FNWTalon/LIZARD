// F3 SELECT in GLSL, the WGSL bank.mjs (archived) SELECT statement for statement (that file says what it keeps, and why the list's
// order is the lanes' appends): the strongest cap peaks of a frame from its tiles' slots, one workgroup a frame.
// ai: Bindings as the WGSL's: 0 frames, 1 raw, 2 peaks, 3 counts, 4 the uniform (slots, cap, tau, p0). hist, used and
// ai: used2 are read before any lane writes them (the histogram's counts, the appends' places), so they are zeroed by
// ai: their initializers (GL_EXT_null_initializer, the same OpConstantNull naga gives every workgroup variable); cut and
// ai: above are written by lane 0 before the barriers that precede their reads.
// ai: The atomics are common.mjs's, at naga's scopes (wgAdd, wgLoad, wgStore, devAdd, devStore).
import { VERSION, ATOMICS, NULL_INIT, DIMS, wgAdd, wgLoad, wgStore, devAdd, devStore } from "./common.mjs";

export const SELECT = VERSION + ATOMICS + NULL_INIT + DIMS + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Raw { vec4 raw[]; };
layout(set = 0, binding = 2, std430) buffer Peaks { vec4 peaks[]; };
layout(set = 0, binding = 3, std430) buffer Counts { uint counts[]; };
// ai: slots: a frame's over every level; p0 the first a round's levels hold
layout(set = 0, binding = 4, std140) uniform Sel { uint slots; uint cap; float tau; uint p0; } P;
shared uint hist[64] = {};
shared uint cut;
shared uint above;
shared uint used = {};
shared uint used2 = {};
uint bin(float v) { return uint(clamp(floor(8.0 * log2(v / P.tau)), 0.0, 63.0)); }
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  // ai: a frame record's word (read-only storage), so the whole group returns before any barrier
  if (frames[f].valid != 1u) { return; }
  uint base = f * P.slots;
  for (uint i = P.p0 + li; i < P.slots; i += 256u) {
    float v = raw[base + i].w;
    if (v > 0.0) { ${wgAdd("hist[bin(v)]", "1u")}; }
  }
  barrier();
  if (li == 0u) {
    uint total = 0u;
    int b = 63;
    while (true) {
      uint h = ${wgLoad("hist[b]")};
      if (total + h >= P.cap || b == 0) { break; }
      total += h;
      b--;
    }
    cut = uint(b);
    above = total;
  }
  // ai: WGSL's workgroupUniformLoad(&cut): a barrier, the load, a barrier
  barrier();
  uint t = cut;
  barrier();
  // ai: One pass: everything above the cut bin into the list's first 'above' places (fewer than the cap by
  // ai: construction), the cut bin after them while the cap has room, each by its own count, so no weak peak takes a
  // ai: strong one's place.
  uint a0 = above;
  for (uint i = P.p0 + li; i < P.slots; i += 256u) {
    vec4 e = raw[base + i];
    if (e.w > 0.0) {
      uint bb = bin(e.w);
      if (bb > t) { peaks[f * P.cap + ${wgAdd("used", "1u")}] = e; }
      else if (bb == t) { uint at = a0 + ${wgAdd("used2", "1u")}; if (at < P.cap) { peaks[f * P.cap + at] = e; } }
    }
  }
  barrier();
  if (li == 0u) { ${devStore("counts[f * 16u + 2u]", `min(${wgLoad("used")} + ${wgLoad("used2")}, P.cap)`)}; }
}
`;
