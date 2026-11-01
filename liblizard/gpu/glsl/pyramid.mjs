// The pyramid in GLSL, the WGSL pyramid.mjs (archived) PYRAMID's arithmetic and layout statement for statement (its header says
// what the kernel does): level l + 1 the 2 x 2 mean of level l, levels 1 to 4 from one 64 x 64 block of a frame's
// layer a workgroup, each mean summed from 0 in the same order, so the levels are the WGSL's bit for bit.
// ai: Bindings as the WGSL's: 0 the frame texture (r8unorm 2D array, read by texelFetch without a sampler,
// ai: GL_EXT_samplerless_texture_functions: SPIR-V's OpImageFetch on a sampled image, as naga writes textureLoad),
// ai: 1 the frames, 2 to 5 levels 1 to 4 (f32), 6 the levels' (stride, rows). Every workgroup variable is written
// ai: before the barrier that precedes its reads, so none is zeroed. Dispatched (ceil(W / 64), ceil(H / 64), frames).
import { VERSION, SAMPLERLESS, DIMS } from "./common.mjs";
import { PYRAMID_BLOCK } from "../plan/pyramid.mjs";

if (PYRAMID_BLOCK !== 64) throw new Error(`the GLSL pyramid is written for a block of 64, not ${PYRAMID_BLOCK}`);

// ai: The mean of the 2 x 2 under pixel (x, y) of level l + 1 from tile t (side s) of level l, the block's own pixel
// ai: (lx, ly) of it: the right column and the lower row clamp onto the pixel beside them past the frame's edge (w x
// ai: h, level l's).
const down = (t, s, w, h) => /* glsl */ `
    uint c0 = 2u * lx; uint c1 = 2u * x + 1u >= ${w} ? 2u * lx : 2u * lx + 1u;
    uint r0 = 2u * ly * ${s}u; uint r1 = 2u * y + 1u >= ${h} ? 2u * ly * ${s}u : (2u * ly + 1u) * ${s}u;
    float v = mean4(${t}[r0 + c0], ${t}[r0 + c1], ${t}[r1 + c0], ${t}[r1 + c1]);`;

export const PYRAMID = VERSION + SAMPLERLESS + DIMS + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform texture2DArray img;
layout(set = 0, binding = 1, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 2, std430) buffer Level1 { float lv1[]; };
layout(set = 0, binding = 3, std430) buffer Level2 { float lv2[]; };
layout(set = 0, binding = 4, std430) buffer Level3 { float lv3[]; };
layout(set = 0, binding = 5, std430) buffer Level4 { float lv4[]; };
layout(set = 0, binding = 6, std140) uniform LevelDims { uvec4 d[5]; } LD;
shared float t1[1024];   // ai: the block's level 1, 32 x 32
shared float t2[256];    // ai: level 2, 16 x 16
shared float t3[64];     // ai: level 3, 8 x 8
uint dimAt(uint v, uint level) { return (v + (1u << level) - 1u) >> level; }
float mean4(float tl, float tr, float bl, float br) {
  float s = 0.0;
  s += tl;
  s += tr;
  s += bl;
  s += br;
  return 0.25 * s;
}
float tex(uint f, uint x, uint y, uint w, uint h) {
  return texelFetch(img, ivec3(int(min(x, w - 1u)), int(min(y, h - 1u)), int(f)), 0).r;
}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.z;
  Frame F = frames[f];
  bool live = F.valid != 0u;
  uint w1 = dimAt(F.w, 1u); uint h1 = dimAt(F.h, 1u);
  uint w2 = dimAt(F.w, 2u); uint h2 = dimAt(F.h, 2u);
  uint w3 = dimAt(F.w, 3u); uint h3 = dimAt(F.h, 3u);
  uint w4 = dimAt(F.w, 4u); uint h4 = dimAt(F.h, 4u);
  // ai: Level 1: pixel (lx, ly) of the block's 32 x 32 at i = li + 256 k, a row of 32 every 32 lanes.
  for (uint k = 0u; k < 4u; k++) {
    uint i = li + 256u * k;
    uint lx = i & 31u;
    uint ly = i >> 5u;
    uint x = 32u * wg.x + lx;
    uint y = 32u * wg.y + ly;
    float v = 0.0;
    if (live) {
      float p00 = tex(f, 2u * x, 2u * y, F.w, F.h);
      float p01 = tex(f, 2u * x + 1u, 2u * y, F.w, F.h);
      float p10 = tex(f, 2u * x, 2u * y + 1u, F.w, F.h);
      float p11 = tex(f, 2u * x + 1u, 2u * y + 1u, F.w, F.h);
      v = mean4(p00, p01, p10, p11);
      if (x < w1 && y < h1) { lv1[(f * LD.d[1].y + y) * LD.d[1].x + x] = v; }
    }
    t1[i] = v;
  }
  barrier();
  {
    uint lx = li & 15u;
    uint ly = li >> 4u;
    uint x = 16u * wg.x + lx;
    uint y = 16u * wg.y + ly;${down("t1", 32, "w1", "h1")}
    if (live && x < w2 && y < h2) { lv2[(f * LD.d[2].y + y) * LD.d[2].x + x] = v; }
    t2[li] = v;
  }
  barrier();
  if (li < 64u) {
    uint lx = li & 7u;
    uint ly = li >> 3u;
    uint x = 8u * wg.x + lx;
    uint y = 8u * wg.y + ly;${down("t2", 16, "w2", "h2")}
    if (live && x < w3 && y < h3) { lv3[(f * LD.d[3].y + y) * LD.d[3].x + x] = v; }
    t3[li] = v;
  }
  barrier();
  if (li < 16u) {
    uint lx = li & 3u;
    uint ly = li >> 2u;
    uint x = 4u * wg.x + lx;
    uint y = 4u * wg.y + ly;${down("t3", 8, "w3", "h3")}
    if (live && x < w4 && y < h4) { lv4[(f * LD.d[4].y + y) * LD.d[4].x + x] = v; }
  }
}
`;
