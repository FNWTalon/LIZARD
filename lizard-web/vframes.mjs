// ai: The camera's frames as VideoFrames for the receiver's CPU pool (recv.mjs sendPoolFrame): the page hands a worker the
// ai: frame with the crop it would have grabbed, and the worker copies the crop to luma (recv-worker.mjs lumaOf), so no
// ai: canvas is drawn or read back on the page. frameTap is the page's half, lumaOf the worker's. No top-level await:
// ai: workers import it.

// ai: One channel of a crop at three points of each corner, (12, 12), (3, 12) and (12, 3) pixels in, for
// ai: lizard-web/check_rates.mjs's crop check (recv.html?verify): its clip has a marker at each corner of the true centre
// ai: square that starts 4 pixels in, covering the first point and stopping short of the other two, so a crop off by a
// ai: pixel either way puts some corner's background point on its marker. stride 4 reads the red channel of RGBA, 1 luma.
export function cornerProbe(px, w, h, stride) {
  const at = (x, y) => px[stride * (y * w + x)], out = [];
  for (const [cx, cy] of [[0, 0], [1, 0], [1, 1], [0, 1]]) for (const [dx, dy] of [[12, 12], [3, 12], [12, 3]]) out.push(at(cx ? w - 1 - dx : dx, cy ? h - 1 - dy : dy));
  return out;
}

// ai: The camera's frames as VideoFrames: the newest off the track through a MediaStreamTrackProcessor where the
// ai: browser has one on the page (Chrome), else the <video>'s current frame (new VideoFrame(video): Safari). take()
// ai: hands over a frame the caller must transfer or close, or null when the track has delivered none since the last
// ai: take. Only the newest is held, an older one closed as the next arrives: every open frame holds a camera buffer.
// ai: stat, since the last resetStat(): frames the track delivered, those closed unseen (a second came before a take),
// ai: and the largest gap in ms between two deliveries; a tap on the <video> delivers nothing on its own, so its stat
// ai: stays at zero.
export function frameTap(track, video) {
  const stat = { delivered: 0, closedUnseen: 0, gapMax: 0 };
  const resetStat = () => { stat.delivered = 0; stat.closedUnseen = 0; stat.gapMax = 0; };
  if (typeof MediaStreamTrackProcessor !== "function") return { kind: "video", stat, resetStat, take: () => { try { return new VideoFrame(video); } catch { return null; } }, stop() {} };
  const reader = new MediaStreamTrackProcessor({ track }).readable.getReader();
  let newest = null, stopped = false, lastAt = 0;
  const tap = {
    kind: "track",
    stat, resetStat,
    take() { const f = newest; newest = null; return f; },
    // ai: Cancelling the processor's stream lets the track go on to the <video>; it does not stop the camera.
    stop() { stopped = true; newest?.close(); newest = null; reader.cancel().catch(() => {}); },
  };
  (async () => {
    for (;;) {
      const { value, done } = await reader.read().catch(() => ({ done: true }));
      if (done) return;
      if (stopped) { value.close(); return; }
      const now = performance.now();
      stat.delivered++;
      if (lastAt) stat.gapMax = Math.max(stat.gapMax, now - lastAt);
      lastAt = now;
      if (newest) { newest.close(); stat.closedUnseen++; }
      newest = value;
    }
  })();
  return tap;
}

// ai: The crop's luma out of a VideoFrame, for a pool worker: every frame, closed here whatever happens. The crop (x, y,
// ai: w, h) is in the frame's display pixels, the space the page works in (the video's). copyTo's rect is in coded
// ai: pixels, and a camera frame need not be 1:1 with its display size: the
// ai: S26 Ultra at 2560 x 1440 hands over a coded 1848 x 4000 frame whose visible rect (0, 356, 1848, 3286) is displayed
// ai: as 1440 x 2560 (at 1920 x 1080 the frame is 1080 x 1920, 1:1), and a crop copied as a coded rect took the wrong rows
// ai: at 1.28x (the void 1440 recording of 2026-09-25 06-50-02). So the crop is mapped into the visible rect and, where
// ai: the frame is scaled, sampled bilinearly at the display pixels' centres.
// ai: A planar YUV frame keeps its luma in plane 0: copyTo takes the crop as a rect (even coordinates on a subsampled
// ai: format) but refuses a layout of fewer planes than the format has (Chrome 153: "Expected 3 planes, found 1"), so
// ai: every plane is copied (1.5 bytes a pixel on I420, 0.3 ms at a 720 crop) and plane 0 is the luma, a view when its
// ai: rows are packed, as Chrome packs them. An RGB frame is converted with the rig's formula (src/acquire.c), so
// ai: every path hands the codec the same quantity. Any other format throws.
// ai: PLANAR_Y8: the pixel formats whose plane 0 is the luma at a byte a sample (the S26 Ultra's camera and headless
// ai: Chrome's fake one give I420; not the 10 and 12 bit forms, two bytes a sample); RGB_RED: the 8-bit RGB ones, with
// ai: the byte that holds red.
const PLANAR_Y8 = /^(I4(20|22|44)A?|NV12|NV21)$/, RGB_RED = { RGBA: 0, RGBX: 0, BGRA: 2, BGRX: 2 };
export async function lumaOf(vf, x, y, w, h) {
  const format = vf.format ?? "unknown";
  try {
    if (!PLANAR_Y8.test(format) && !(format in RGB_RED)) throw new Error(`${format}: not a pixel format this worker reads`);
    const vr = vf.visibleRect, sx = vr.width / vf.displayWidth, sy = vr.height / vf.displayHeight, scaled = sx !== 1 || sy !== 1;
    // ai: Where display pixel (x, y)'s centre lands in coded pixels, less the half pixel bilinear taps work from: the
    // ai: crop's first sample. The coded rect copied covers every sample and, when scaled, the tap beyond the last, at
    // ai: even coordinates inside the coded frame.
    const u0 = vr.x + (x + 0.5) * sx - 0.5, v0 = vr.y + (y + 0.5) * sy - 0.5;
    const cx0 = Math.max(0, Math.floor(u0)) & ~1, cy0 = Math.max(0, Math.floor(v0)) & ~1;
    const cx1 = Math.min(vf.codedWidth, (Math.ceil(u0 + (w - 1) * sx) + (scaled ? 2 : 1) + 1) & ~1), cy1 = Math.min(vf.codedHeight, (Math.ceil(v0 + (h - 1) * sy) + (scaled ? 2 : 1) + 1) & ~1);
    const rect = { x: cx0, y: cy0, width: cx1 - cx0, height: cy1 - cy0 };
    const buf = new ArrayBuffer(vf.allocationSize({ rect })), [p0] = await vf.copyTo(buf, { rect });
    // ai: The copied rect's luma plane: bytes, the plane's offset in them and its row stride.
    let plane = new Uint8Array(buf), po = p0.offset, ps = p0.stride;
    if (format in RGB_RED) {
      const R = RGB_RED[format], B = 2 - R, pw = rect.width, L = new Uint8Array(pw * rect.height);
      for (let r = 0; r < rect.height; r++) for (let c = 0, i = r * pw, j = po + r * ps; c < pw; c++, i++, j += 4) L[i] = (77 * plane[j + R] + 150 * plane[j + 1] + 29 * plane[j + B] + 128) >> 8;
      plane = L; po = 0; ps = pw;
    }
    if (scaled) return { luma: resample(plane, po, ps, rect.width, rect.height, u0 - cx0, sx, v0 - cy0, sy, w, h), format };
    const ox = Math.round(u0) - cx0, oy = Math.round(v0) - cy0;
    if (ox === 0 && oy === 0 && ps === w) return { luma: new Uint8Array(plane.buffer, plane.byteOffset + po, w * h), format };
    const luma = new Uint8Array(w * h);
    for (let r = 0; r < h; r++) luma.set(plane.subarray(po + (oy + r) * ps + ox, po + (oy + r) * ps + ox + w), r * w);
    return { luma, format };
  } finally { vf.close(); }
}

// ai: w x h samples of a luma plane (pw x ph at offset po, rows ps apart), output (i, j) taken bilinearly at plane position
// ai: (u0 + i du, v0 + j dv), the position of a pixel's top left tap, so a plane pixel's centre is at k + 0.5. Taps are
// ai: clamped to the plane. Weights are 8-bit fixed point: 2.07 M samples at a 1440 crop, so no floats in the inner loop.
function resample(plane, po, ps, pw, ph, u0, du, v0, dv, w, h) {
  const out = new Uint8Array(w * h), kx = new Int32Array(w), fx = new Int32Array(w);
  for (let i = 0; i < w; i++) { const u = Math.min(Math.max(u0 + i * du, 0), pw - 1), k = Math.min(Math.floor(u), pw - 2); kx[i] = k; fx[i] = Math.round((u - k) * 256); }
  for (let j = 0; j < h; j++) {
    const v = Math.min(Math.max(v0 + j * dv, 0), ph - 1), l = Math.min(Math.floor(v), ph - 2), fy = Math.round((v - l) * 256), gy = 256 - fy;
    const r0 = po + l * ps, o = j * w;
    for (let i = 0; i < w; i++) {
      const k = r0 + kx[i], f = fx[i], g = 256 - f;
      const top = plane[k] * g + plane[k + 1] * f, bot = plane[k + ps] * g + plane[k + ps + 1] * f;
      out[o + i] = (top * gy + bot * fy + 32768) >> 16;
    }
  }
  return out;
}
