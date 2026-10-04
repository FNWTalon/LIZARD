// ai: F0, the ingest: a camera frame on the device (a GPUExternalTexture imported from a VideoFrame, no copy through
// ai: the CPU) to a batch's r8unorm layer, cropped, as luma by the rig's integer formula on 8-bit values
// ai: (77 R + 150 G + 29 B + 128) >> 8, as lizard-web/glgrab.mjs and the canvas grab compute it. A render pass, because
// ai: r8unorm is not a storage format in core WebGPU: one triangle over the target, a fragment a luma pixel.
// ai: The crop's origin rides in the draw's first instance (x + 65536 y), so the pass needs no buffer; the viewport,
// ai: set to the crop's size, bounds it.
export const INGEST = /* wgsl */ `
@group(0) @binding(0) var src: texture_external;

struct Out { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) origin: vec2u }

@vertex fn vs(@builtin(vertex_index) i: u32, @builtin(instance_index) o: u32) -> Out {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return Out(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2u(o & 0xffffu, o >> 16u));
}

@fragment fn fs(v: Out) -> @location(0) vec4f {
  // ai: The target's row 0 is the crop's row 0: WebGPU's framebuffer origin is the top left, as the frame's is.
  let c = textureLoad(src, v.origin + vec2u(v.pos.xy));
  let q = vec3u(floor(saturate(c.rgb) * 255.0 + 0.5));
  return vec4f(f32((77u * q.r + 150u * q.g + 29u * q.b + 128u) >> 8u) / 255.0, 0.0, 0.0, 1.0);
}
`;
