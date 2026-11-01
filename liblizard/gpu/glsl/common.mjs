// GLSL shared by the native decoder's GLSL stages (../../gen/gen.mjs compiles them with glslang to SPIR-V 1.3; the
// host loads them as it loads naga's). A stage given here in GLSL kept its WGSL in ../wgsl/ as a twin, read only
// by a WebGPU device, until 2026-10-31 (archived). GLSL here has no includes either, so a stage's source is these pieces joined in JS, as the
// WGSL's are.
// ai: Every source begins with VERSION, then its extensions, then its pieces. Bindings are set 0, binding k as the
// ai: WGSL's @group(0) @binding(k), against the host's bind group layout (the JS host's createBindGroupLayout):
// ai: storage arrays std430 (WGSL's storage layout), uniform blocks std140 (WGSL's uniform layout: arrays of vec4
// ai: and scalars at their WGSL offsets; an array of scalars or vec2 in a uniform would differ and is not used).
// ai: Workgroup memory is not zeroed unless a stage asks (GL_EXT_null_initializer, `= {}`), where WGSL zeroes every
// ai: workgroup variable: a stage that reads before it writes says so.

export const VERSION = "#version 460\n";

// ai: The extensions a stage may ask for, each by one name: VERSION, then the stage's extensions, then its pieces.
export const SAMPLERLESS = "#extension GL_EXT_samplerless_texture_functions : require\n";
export const NULL_INIT = "#extension GL_EXT_null_initializer : require\n";
// ai: atomicLoad and atomicStore (OpAtomicLoad, OpAtomicStore) and an atomic's scope (wgAdd and the rest below); it
// ai: changes no memory model and asks no device feature.
export const ATOMICS = "#extension GL_KHR_memory_scope_semantics : require\n";
export const F16 = "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
export const STORAGE16 = "#extension GL_EXT_shader_16bit_storage : require\n";
// ai: int16_t and uint16_t (SPIR-V Int16, the device's shaderInt16): in workgroup memory and registers that feature
// ai: alone (the 16-bit storage features cover buffers, uniforms, push constants and interfaces, none of which such a
// ai: module declares 16-bit); a module that names it is built only where the device has shaderInt16 (its pipeline's
// ai: `needs`, ../../gen/gen.mjs needsOf, ../../core/dec/setup.cpp)
export const INT16 = "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
export const DOT = "#extension GL_EXT_integer_dot_product : require\n";
export const SHUFFLE = "#extension GL_KHR_shader_subgroup_shuffle : require\n";
// ai: [[unroll]] and [[dont_unroll]] on a loop (GL_EXT_control_flow_attributes)
export const UNROLL = "#extension GL_EXT_control_flow_attributes : require\n";

// ai: A frame of a batch, the WGSL's Frame (the WGSL common.mjs (archived) DIMS): (w, h, axis, valid), 16 B.
export const DIMS = /* glsl */ `
struct Frame { uint w; uint h; uint size; uint valid; };
`;

// ai: WGSL's float-to-integer conversions saturate (naga clamps in f32 before each OpConvertFTo*: to
// ai: [-2147483648, 2147483520] for i32, [0, 4294967040] for u32), where GLSL's int() and uint() of a float out of
// ai: range are undefined: a stage ported from WGSL converts every float through these, so a degenerate value (a
// ai: frame's map gone to infinity) reads as the WGSL's does. glslang drops the overloads a stage never calls.
export const CONVERT = /* glsl */ `
int f2i(float x) { return int(clamp(x, -2147483648.0, 2147483520.0)); }
ivec2 f2i(vec2 x) { return ivec2(clamp(x, vec2(-2147483648.0), vec2(2147483520.0))); }
ivec4 f2i(vec4 x) { return ivec4(clamp(x, vec4(-2147483648.0), vec4(2147483520.0))); }
uint f2u(float x) { return uint(clamp(x, 0.0, 4294967040.0)); }
uvec2 f2u(vec2 x) { return uvec2(clamp(x, vec2(0.0), vec2(4294967040.0))); }
`;

// ai: Level 0's bilinear read (the WGSL common.mjs (archived) LUM0, its arithmetic and clamps): pixel centres at i + 0.5,
// ai: clamped to the frame's own edge. A stage including it binds the frame texture as `img` (texture2DArray,
// ai: SAMPLERLESS) and includes CONVERT.
export const LUM0 = /* glsl */ `
float lum0(uint f, int x, int y, int w, int h) {
  return texelFetch(img, ivec3(clamp(x, 0, w - 1), clamp(y, 0, h - 1), int(f)), 0).r;
}
float bilin0(uint f, vec2 p, int w, int h) {
  vec2 q = p - vec2(0.5);
  vec2 q0 = floor(q);
  vec2 a = q - q0;
  ivec2 i = f2i(q0);
  float t = mix(lum0(f, i.x, i.y, w, h), lum0(f, i.x + 1, i.y, w, h), a.x);
  float b = mix(lum0(f, i.x, i.y + 1, w, h), lum0(f, i.x + 1, i.y + 1, w, h), a.x);
  return mix(t, b, a.y);
}
`;

// ai: Level 0's bilinear read by the texture unit (LUM0's bilin0 in one fetch): the frame texture through the device's
// ai: linear sampler (clamp to the edge, normalised coordinates; natively the layout's immutable sampler, core/wg
// ai: Device::linearSampler), the weights in the device's fixed point (8 bits a texel on the 4090, RADV and the
// ai: Adreno 840). p is clamped to the frame's outer texel centres, [0.5, w - 0.5] x [0.5, h - 0.5]: there the far
// ai: texel's weight is exactly 0, so a read past the frame takes its edge texel as lum0's clamp does (the layer may
// ai: be wider and taller than the frame; the sampler clamps only at the layer's edge). hi = (w, h) - 0.5 and inv =
// ai: one over the layer's size, the caller's, made once. A stage including it binds the frame texture as `img`
// ai: (texture2DArray) and the sampler at the binding given.
export const LUMH = (binding) => /* glsl */ `
layout(set = 0, binding = ${binding}) uniform sampler smp;
float bilinH(uint f, vec2 p, vec2 hi, vec2 inv) {
  return textureLod(sampler2DArray(img, smp), vec3(clamp(p, vec2(0.5), hi) * inv, float(f)), 0.0).r;
}
`;
// ai: hi and inv for bilinH, a frame of w x h in the bound layer (int w, h)
export const lumhHead = (w, h) => `  vec2 hhi = vec2(float(${w}), float(${h})) - vec2(0.5);
  vec2 hinv = vec2(1.0) / vec2(textureSize(img, 0).xy);
`;

// ai: A frame's map from module coordinates to image pixels (the WGSL common.mjs (archived) MAP, statement for statement): 16
// ai: floats, the homography, then the radial term about a centre (k1 = 0 a plain homography). A buffer of maps is
// ai: `float maps[][16]` (the WGSL's array<array<f32, 16>>, 64 B a map in std430).
export const MAP = /* glsl */ `
vec2 toImage(float M[16], vec2 m) {
  float w = M[6] * m.x + M[7] * m.y + M[8];
  vec2 u = vec2(M[0] * m.x + M[1] * m.y + M[2], M[3] * m.x + M[4] * m.y + M[5]) / w;
  if (M[9] == 0.0) { return u; }
  vec2 c = vec2(M[10], M[11]);
  vec2 e = u - c;
  float g = M[9] * dot(e, e) * M[12];
  float s = 1.0;
  for (int i = 0; i < 3; i++) { float gs = g * s * s; s = s - (s * (1.0 + gs) - 1.0) / (1.0 + 3.0 * gs); }
  return c + e * s;
}
`;

// ai: The pyramid's levels and the frames, bindings 0 to 6 (the WGSL finder.mjs (archived) LEVEL_BINDINGS; register.mjs's level
// ai: reader binds them too). LEVELS reads every level by level-0 coordinates (the WGSL finder.mjs (archived) LEVELS, which the
// ai: classifier copies); a stage including LEVELS includes CONVERT, DIMS and SAMPLERLESS.
export const LEVEL_BINDINGS = /* glsl */ `
layout(set = 0, binding = 0) uniform texture2DArray img;
layout(set = 0, binding = 1, std430) readonly buffer Level1 { float lv1[]; };
layout(set = 0, binding = 2, std430) readonly buffer Level2 { float lv2[]; };
layout(set = 0, binding = 3, std430) readonly buffer Level3 { float lv3[]; };
layout(set = 0, binding = 4, std430) readonly buffer Level4 { float lv4[]; };
layout(set = 0, binding = 5, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 6, std140) uniform LevelDims { uvec4 d[5]; } LD;
`;
export const LEVELS = /* glsl */ `
float pix(uint l, uint f, int x, int y) {
  Frame F = frames[f];
  int w = int((F.w + (1u << l) - 1u) >> l);
  int h = int((F.h + (1u << l) - 1u) >> l);
  int xc = clamp(x, 0, w - 1);
  int yc = clamp(y, 0, h - 1);
  if (l == 0u) { return texelFetch(img, ivec3(xc, yc, int(f)), 0).r; }
  uvec4 s = LD.d[l];
  uint i = (f * s.y + uint(yc)) * s.x + uint(xc);
  switch (l) {
    case 1u: { return lv1[i]; }
    case 2u: { return lv2[i]; }
    case 3u: { return lv3[i]; }
    default: { return lv4[i]; }
  }
}
float look(uint l, uint f, vec2 p) {
  vec2 q = p / float(1u << l) - vec2(0.5);
  vec2 q0 = floor(q);
  vec2 a = q - q0;
  ivec2 i = f2i(q0);
  return mix(mix(pix(l, f, i.x, i.y), pix(l, f, i.x + 1, i.y), a.x), mix(pix(l, f, i.x, i.y + 1), pix(l, f, i.x + 1, i.y + 1), a.x), a.y);
}
bool onFrame(uint f, vec2 p) { Frame F = frames[f]; return p.x >= 0.0 && p.y >= 0.0 && p.x < float(F.w) && p.y < float(F.h); }
uint levelFor(float u) { return f2u(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0)); }
vec2 dir(float a) { return vec2(cos(a), sin(a)); }
`;

// ai: JS helpers for the templates, WGSL's built-ins as naga writes them:
// ai: lit(x) a JS number as a GLSL float literal (the WGSL writes some as integers, which WGSL converts and GLSL may
// ai: not); zeros(n) or zeros(n, m) a GLSL zero of a float array (WGSL's array<f32, n>(), a function variable's zero);
// ai: uniformLoad(decl, src) WGSL's workgroupUniformLoad into a new variable (a barrier, the load, a barrier);
// ai: STORAGE_BARRIER WGSL's storageBarrier(); the atomics at naga's scope, relaxed (ATOMICS): workgroup memory at
// ai: workgroup scope (GLSL's plain atomicAdd there is device scope), storage at device scope; wgAdd(x, v) is WGSL's
// ai: atomicAdd(&x, v) on a workgroup variable, and so on.
// ai: WGSL's pack4xI8 as the low bytes shifted together (gen/gen.mjs's liz_pack4xI8 rewrite for naga: the Adreno 840
// ai: got naga's 8-bit vector form wrong), a GLSL function for the int8 kernels.
export const PACK4X_I8 = "uint pack4xI8(ivec4 v) { uvec4 u = uvec4(v) & uvec4(0xffu); return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u); }\n";
export const lit = (x) => (Number.isInteger(x) ? `${x}.0` : String(x));
export const zeros = (n, m) => (m === undefined ? `float[${n}](${Array(n).fill("0.0").join(", ")})` : `float[${n}][${m}](${Array(n).fill(zeros(m)).join(", ")})`);
export const uniformLoad = (decl, src) => `barrier(); ${decl} = ${src}; barrier();`;
export const STORAGE_BARRIER = "memoryBarrierBuffer(); barrier();";
export const wgAdd = (x, v) => `atomicAdd(${x}, ${v}, gl_ScopeWorkgroup, 0, 0)`;
export const wgLoad = (x) => `atomicLoad(${x}, gl_ScopeWorkgroup, 0, 0)`;
export const wgStore = (x, v) => `atomicStore(${x}, ${v}, gl_ScopeWorkgroup, 0, 0)`;
export const wgXor = (x, v) => `atomicXor(${x}, ${v}, gl_ScopeWorkgroup, 0, 0)`;
export const devAdd = (x, v) => `atomicAdd(${x}, ${v})`;
export const devLoad = (x) => `atomicLoad(${x}, gl_ScopeDevice, 0, 0)`;
export const devStore = (x, v) => `atomicStore(${x}, ${v}, gl_ScopeDevice, 0, 0)`;
