// Patches for the classifier, cut in JS exactly as gpu/wgsl/finder.mjs reads the image: a pyramid of 2 x 2 means,
// bilinear with pixel centres at k + 0.5, coordinates in level-0 pixels (scripts/gpu/cnn/README.md, "The dataset").
export const PATCH = 32, KAPPA = 1.8;

export function pyramid(img, w, h) {
  const levels = [{ w, h, px: img }];
  for (let l = 1; l <= 4; l++) {
    const p = levels[l - 1], pw = p.w, ph = p.h, lw = Math.ceil(pw / 2), lh = Math.ceil(ph / 2), out = new Float32Array(lw * lh);
    for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) {
      const x0 = 2 * x, y0 = 2 * y, x1 = Math.min(x0 + 1, pw - 1), y1 = Math.min(y0 + 1, ph - 1);
      out[y * lw + x] = 0.25 * (p.px[y0 * pw + x0] + p.px[y0 * pw + x1] + p.px[y1 * pw + x0] + p.px[y1 * pw + x1]);
    }
    levels.push({ w: lw, h: lh, px: out });
  }
  return levels;
}

export const levelFor = (u) => Math.min(4, Math.max(0, Math.floor(Math.log2(Math.max(u, 1) / 1.5))));

// Bilinear on level l at level-0 point (x, y); level 0 is u8 0..255, the rest f32 in the same units.
export function look(pyr, l, x, y) {
  const L = pyr[l], q = 1 << l, qx = x / q - 0.5, qy = y / q - 0.5, x0 = Math.floor(qx), y0 = Math.floor(qy), fx = qx - x0, fy = qy - y0;
  const at = (i, j) => L.px[Math.min(L.h - 1, Math.max(0, j)) * L.w + Math.min(L.w - 1, Math.max(0, i))];
  const t = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx, b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx;
  return t + (b - t) * fy;
}

// The 32 x 32 patch round peak (x, y) of scale sigma, one module a sample, as u8.
export function cutPatch(pyr, x, y, sigma, out = new Uint8Array(PATCH * PATCH)) {
  const u = sigma / KAPPA, l = levelFor(u);
  for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) {
    const v = look(pyr, l, x + (i - 15.5) * u, y + (j - 15.5) * u);
    out[j * PATCH + i] = Math.max(0, Math.min(255, Math.round(v)));
  }
  return out;
}

// A homography from four point pairs (module coordinates to image pixels), row major with h[8] = 1.
export function homography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  for (let c = 0; c < 8; c++) {
    let p = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      if (!f) continue;
      for (let k = c; k < 8; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  return [...b.map((v, i) => v / A[i][i]), 1];
}
export const applyH = (H, x, y) => { const w = H[6] * x + H[7] * y + H[8]; return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w]; };

// The four marks of a symbol M modules a side whose map from module coordinates is `at(mx, my)`: gapped and
// merged centres in image pixels, the module size there, and the outward angle from the symbol's centre.
export function marksOf(at, M) {
  const [cx, cy] = at(M / 2, M / 2);
  return [[0, 0], [1, 0], [1, 1], [0, 1]].map(([ax, ay]) => {
    const c = (d) => at(ax ? M - d : d, ay ? M - d : d);
    const [x, y] = c(8), [mx, my] = c(6), gx = ax ? M - 8 : 8, gy = ay ? M - 8 : 8;
    const [x1, y1] = at(gx + 1, gy), [x2, y2] = at(gx, gy + 1);
    return { x, y, mx, my, u: (Math.hypot(x1 - x, y1 - y) + Math.hypot(x2 - x, y2 - y)) / 2, theta: Math.atan2(y - cy, x - cx) };
  });
}

// Label a frame's peaks (4 floats each: x, y, sigma, response) against its marks. Returns per peak
// [is_mark, cos t, sin t, dlogu, form, weight].
export function labelPeaks(peaks, marks) {
  const n = peaks.length / 4, out = new Float32Array(6 * n);
  for (let k = 0; k < n; k++) {
    const px = peaks[4 * k], py = peaks[4 * k + 1], u = peaks[4 * k + 2] / KAPPA;
    let best = null, bd = 1e9;
    for (const m of marks) {
      // A mark projected near the camera's horizon has a module of thousands of pixels and a tolerance that covers
      // the frame: two such marks labelled 2,233 peaks as marks in one export (2026-09-24). 128 px is 2.5x the widest
      // module a scene draws.
      if (!(m.u > 0 && m.u <= 128) || !Number.isFinite(m.x + m.y + m.mx + m.my)) continue;
      const tol = Math.max(1.5, 1.25 * m.u), dg = Math.hypot(px - m.x, py - m.y), dm = Math.hypot(px - m.mx, py - m.my), d = Math.min(dg, dm);
      if (d <= tol && d < bd) { bd = d; best = { m, form: dm < dg ? 1 : 0 }; }
    }
    out[6 * k + 5] = 1;
    if (!best) continue;
    out[6 * k] = 1; out[6 * k + 1] = Math.cos(best.m.theta); out[6 * k + 2] = Math.sin(best.m.theta);
    out[6 * k + 3] = Math.log(best.m.u / u); out[6 * k + 4] = best.form;
  }
  return out;
}
