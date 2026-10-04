// The camera frame becomes luma on the GPU and comes back through a fence, instead of drawImage into a
// software canvas and a getImageData four times the size of what the decoder wants.
//
// The old path costs four bytes a pixel at every boundary to deliver one: the 2D context is
// willReadFrequently, so it is a software canvas and the YUV to RGB conversion runs on the CPU, then
// getImageData copies 2.07 MB at the 720 crop, then the worker hands it to _ob_luma_strip to be made luma
// inside wasm. Here the conversion is the sampler's, the luma is a fragment shader, and 518 KB comes back.
//
// WebGL2 and not WebGPU: Firefox for Android has no WebGPU and every real figure this project has came from
// it, so a WebGPU arm could not be compared against the baseline without first measuring the baseline on
// another browser. WebGL2 is the worse choice if a KERNEL ever moves here, since a fragment-shader FFT
// ping-pongs textures once per butterfly stage with no workgroup memory to hold a row, but no kernel moves.
//
// Four luma values ride in one RGBA8 texel, so the render target is a quarter of the crop's width and
// readPixels hands back the luma scanline already in order, with nothing to unpack on the CPU.

const VERT = `#version 300 es
void main() {
  // One triangle covering the clip square, from gl_VertexID: no attributes, no buffer, nothing to bind.
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D uTex;
uniform ivec2 uOrigin;   // the crop's origin in camera pixels
uniform ivec2 uSize;     // the crop's size in camera pixels
uniform int uPreview;    // 1: draw the crop for the screen. 0: four luma to a texel, for readback.
out vec4 o;

float lumaAt(int x, int y) {
  // Clamped, so a width that is not a multiple of four repeats its last pixel instead of reading past the crop.
  ivec2 p = uOrigin + ivec2(min(x, uSize.x - 1), y);
  vec3 c = texelFetch(uTex, p, 0).rgb * 255.0;
  // The same integer form as _ob_luma_strip, so the two grabs can be compared byte for byte.
  return floor((77.0 * floor(c.r + 0.5) + 150.0 * floor(c.g + 0.5) + 29.0 * floor(c.b + 0.5) + 128.0) / 256.0);
}

void main() {
  ivec2 f = ivec2(gl_FragCoord.xy);
  if (uPreview == 1) {
    // The canvas is the crop's size, so this is one to one. y is turned over because a framebuffer's row 0 is
    // its bottom row and the crop's row 0 is its top.
    o = vec4(texelFetch(uTex, uOrigin + ivec2(f.x, uSize.y - 1 - f.y), 0).rgb, 1.0);
    return;
  }
  // No flip here, and for the opposite reason: readPixels starts at the framebuffer's row 0, and row 0 of the
  // readback has to be row 0 of the crop. The picture in this target is upside down and is never looked at.
  int x = f.x * 4, y = f.y;
  o = vec4(lumaAt(x, y), lumaAt(x + 1, y), lumaAt(x + 2, y), lumaAt(x + 3, y)) / 255.0;
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
  return s;
}

// video: the <video> the camera is playing into. colorSpace: "none" or "default", which picks whether the
// browser is allowed to colour-manage the upload. Limited against full range is the one real precision risk on
// this path, so it is a switch and not a decision made here.
export function makeGlGrab(video, { colorSpace = "none", slots = 3 } = {}) {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", {
    alpha: false, antialias: false, depth: false, stencil: false,
    preserveDrawingBuffer: false, desynchronized: true, powerPreference: "low-power",
  });
  if (!gl) return null;

  let prog, loc, vao, tex, rb, fbo;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? "link");
  } catch { gl.getExtension("WEBGL_lose_context")?.loseContext(); return null; }
  loc = { tex: gl.getUniformLocation(prog, "uTex"), origin: gl.getUniformLocation(prog, "uOrigin"), size: gl.getUniformLocation(prog, "uSize"), preview: gl.getUniformLocation(prog, "uPreview") };
  vao = gl.createVertexArray();   // WebGL2 wants one bound even with no attributes enabled
  tex = gl.createTexture();
  rb = gl.createRenderbuffer();   // written and read back, never sampled, so a renderbuffer and not a texture
  fbo = gl.createFramebuffer();

  gl.bindTexture(gl.TEXTURE_2D, tex);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);            // texelFetch indexes the frame directly; a flip here would move every row
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, colorSpace === "default" ? gl.BROWSER_DEFAULT_WEBGL : gl.NONE);
  gl.pixelStorei(gl.PACK_ALIGNMENT, 1);

  // One per in-flight readback. Three, so a frame never waits on the fence of the frame before it.
  const ring = Array.from({ length: slots }, () => ({ buf: gl.createBuffer(), sync: null, bytes: 0, w: 0, h: 0, stride: 0, tag: 0 }));
  let texW = 0, texH = 0, fboW = 0, fboH = 0, lost = false;
  const stat = { submitted: 0, taken: 0, noSlot: 0, failed: 0, uploads: 0, uploadMs: 0 };
  // The preview and the readback are two passes over ONE camera frame, so the frame goes to the GPU once for
  // both. newFrame() marks a new one; until it is called again, upload() knows the texture already holds it.
  let mark = 0, marked = -1;

  canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); lost = true; });

  function upload() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return false;
    if (marked === mark) return true;
    const t0 = performance.now();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // Reallocating every frame makes some drivers throw the old storage away and take a new one; after the
    // first frame of a size, only the contents change.
    if (vw !== texW || vh !== texH) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video); texW = vw; texH = vh; }
    else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, video);
    marked = mark; stat.uploads++; stat.uploadMs += performance.now() - t0;
    return true;
  }
  function draw(r, preview) {
    gl.useProgram(prog); gl.bindVertexArray(vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(loc.tex, 0);
    gl.uniform2i(loc.origin, r.x, r.y); gl.uniform2i(loc.size, r.w, r.h); gl.uniform1i(loc.preview, preview ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Sizes the readback target if the crop changed, then draws the luma pass into it. Returns the target's
  // width in texels, a quarter of the crop's rounded up, or 0 if it could not be set up.
  function lumaPass(r) {
    const w4 = Math.ceil(r.w / 4);
    if (w4 !== fboW || r.h !== fboH) {
      gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, w4, r.h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { gl.bindFramebuffer(gl.FRAMEBUFFER, null); stat.failed++; return 0; }
      fboW = w4; fboH = r.h;
    } else gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, w4, r.h);
    draw(r, false);
    return w4;
  }
  // The rows come back padded up to a multiple of four. No resolution this rig offers needs the repack.
  function unpack(padded, w, h, stride) {
    if (stride === w) return padded;
    const luma = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) luma.set(padded.subarray(y * stride, y * stride + w), y * w);
    return luma;
  }

  return {
    canvas,
    stat,
    get lost() { return lost; },
    // A new camera frame has arrived; the texture no longer holds it. Cheap, and it does not touch the GPU.
    newFrame: () => { mark++; },

    // Switched away from. Readbacks still in flight belong to a frame the page has stopped caring about, and
    // their tags would land in the file's timing if they were picked up on the way back.
    reset() {
      for (const slot of ring) { if (slot.sync) gl.deleteSync(slot.sync); slot.sync = null; }
    },

    // Free slots in the ring. The caller uses this to decide whether a frame can be started at all.
    slotsFree: () => ring.filter((s) => !s.sync).length,

    // The preview. Same texture and the same crop shader as the readback, so what is on screen and what the
    // decoder gets cannot come from different pixels. They are no longer the same RECTANGLE: the page paints the
    // aiming square and reads back the tracked box inside it, which is strictly narrower, so everything the
    // decoder sees is still on screen. A crop the preview cannot show still cannot exist.
    paint(r) {
      if (lost || !upload()) return false;
      if (canvas.width !== r.w || canvas.height !== r.h) { canvas.width = r.w; canvas.height = r.h; }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, r.w, r.h);
      draw(r, true);
      return true;
    },

    // Luma into a pixel pack buffer, then a fence. readPixels into a bound PACK buffer returns at once; what
    // it costs is paid later, in take(), and only once the GPU says it is done.
    submit(r, tag) {
      if (lost) return false;
      const slot = ring.find((s) => !s.sync);
      if (!slot) { stat.noSlot++; return false; }
      if (!upload()) return false;
      const w4 = lumaPass(r);
      if (!w4) return false;
      const bytes = w4 * 4 * r.h;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, slot.buf);
      if (slot.bytes !== bytes) { gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes, gl.STREAM_READ); slot.bytes = bytes; }
      gl.readPixels(0, 0, w4, r.h, gl.RGBA, gl.UNSIGNED_BYTE, 0);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      Object.assign(slot, { sync: gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0), w: r.w, h: r.h, stride: w4 * 4, tag });
      gl.flush();   // without this the fence can sit unsignalled for as long as the driver likes
      stat.submitted++;
      return true;
    },

    // The oldest finished readback, or null. Never blocks: a zero timeout is a question, not a wait, and a
    // clientWaitSync with a real timeout on this thread would stall the page on the GPU.
    take() {
      if (lost) return null;
      for (const slot of ring) {
        if (!slot.sync) continue;
        const st = gl.clientWaitSync(slot.sync, 0, 0);
        if (st === gl.TIMEOUT_EXPIRED) continue;
        gl.deleteSync(slot.sync); slot.sync = null;
        if (st === gl.WAIT_FAILED) { stat.failed++; continue; }
        const { w, h, stride, tag } = slot;
        const padded = new Uint8Array(stride * h);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, slot.buf);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, padded);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        const luma = unpack(padded, w, h, stride);
        stat.taken++;
        return { luma, w, h, tag };
      }
      return null;
    },

    // Diagnostic only (recv.html?grabcmp). The same luma read back synchronously, so it can be held against the
    // canvas grab of the SAME frame. The pipelined path cannot answer that question: its luma lands a frame or
    // two later, by which time the canvas holds a different picture. Whether the GPU's YUV to RGB agrees with
    // the software canvas's, limited against full range above all, is a property of the browser and the device,
    // so it has to be askable on the phone and not only here.
    readSync(r) {
      marked = -1;   // a diagnostic, and the frame it wants may be one the pipeline already marked as uploaded
      if (lost || !upload()) return null;
      const w4 = lumaPass(r);
      if (!w4) return null;
      const stride = w4 * 4, padded = new Uint8Array(stride * r.h);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      gl.readPixels(0, 0, w4, r.h, gl.RGBA, gl.UNSIGNED_BYTE, padded);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return unpack(padded, r.w, r.h, stride);
    },

    dispose() {
      for (const s of ring) { if (s.sync) gl.deleteSync(s.sync); gl.deleteBuffer(s.buf); }
      gl.deleteFramebuffer(fbo); gl.deleteRenderbuffer(rb); gl.deleteTexture(tex);
      gl.deleteVertexArray(vao); gl.deleteProgram(prog);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
}
