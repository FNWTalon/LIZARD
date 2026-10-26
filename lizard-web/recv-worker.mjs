// Decode worker, one of a pool. It only decodes: which blocks are new and the file's fountain decoder
// are single things, so they live with the page (recv.mjs) and the fountain worker.
// ai: Told nothing (the whole receiver blind since 2026-09-26): LIZARD through sim/phy.mjs makeBlind, each
// ai: frame at the size and version its own word names, its blocks judged from the light (blockJudge: a frame is the
// ai: test stream's when a block carries its id's SHAKE256, and its other blocks are then bad).
// ai:   page to worker: { type: "config", config } ({ version, spec: sim/phy.mjs blindSpec }); { type: "frame", tag,
// ai:     w, h, rgba, delay, prof }, the buffer transferred (the canvas grab); or { type: "frame", tag, frame,
// ai:     x, y, w, h, delay, prof, verify }: the camera's VideoFrame, transferred, and the crop the grab would
// ai:     have read (whole pixels inside its display size), copied to luma here (lumaOf) and then decoded as a luma frame.
// ai:   worker to page: { type: "ready", version, frames, framesWhy } (frames: this worker takes a VideoFrame); one
// ai:     { type: "result", ... } per frame (test and judged: the frame was judged the test stream's and its data blocks
// ai:     held to it; vf: the VideoFrame's pixel format when it came as one; probe: the crop's corners under verify;
// ai:     noFrames: why a VideoFrame could not be read, answered with no decode, on which the page sends the grab for the
// ai:     session); { type: "error", tag, message }.
import { makePhy } from "../liblizard/sim/phy.mjs";
import { heapTop } from "../liblizard/sim/ob.mjs";
import { cornerProbe, lumaOf } from "./vframes.mjs";

let phy = null, cfg = null, chain = Promise.resolve();
let levelTick = 0;
const TAKES_FRAMES = typeof VideoFrame === "function";

// ai: The decoder is rebuilt only when its spec moves (none has since the page's #sizes menu went, 2026-09-29).
async function configure(c) {
  const same = cfg && JSON.stringify(c.spec) === JSON.stringify(cfg.spec);
  cfg = c;
  // ai: setupMs: the decoder's build (the wasm instance and the three ring codecs), for the page's workersFirst.
  const t = performance.now();
  if (!same) { phy?.free(); phy = await makePhy(c.spec); }
  postMessage({ type: "ready", version: c.version, frames: TAKES_FRAMES, framesWhy: TAKES_FRAMES ? null : "no VideoFrame in the worker", setupMs: Math.round(performance.now() - t) });
}

// ai: The answer for a frame that was not decoded: the page waits on every frame it handed out.
const undecoded = (tag, more = {}) => ({ type: "result", tag, found: 0, seen: 0, bad: 0, ms: 0, ids: [], ...more });

async function frame(m) {
  const { w, h, tag, delay, prof } = m;
  // Test hook (recv.html?delay=60): stand in for a slow phone, so the pool's growth can be exercised on a fast machine.
  if (delay) { const end = performance.now() + delay; while (performance.now() < end); }
  // A frame this worker cannot decode is still a frame the page handed it, and the page waits on every one it
  // handed out. Answer with nothing rather than not answering.
  if (!phy) {
    m.frame?.close();
    return postMessage(undecoded(tag));
  }
  // The canvas grab sends RGBA and the decoder converts it to luma inside wasm, vectorized.
  // ai: A VideoFrame is copied to luma here, one byte a pixel, which the decoder takes as it is.
  let isRgba = !m.frame, px, vf = null;
  if (m.frame) {
    try { ({ luma: px, format: vf } = await lumaOf(m.frame, m.x, m.y, w, h)); }
    catch (e) {
      // ai: This frame is lost, as a busy skip is; the page sends the grab from here on.
      return postMessage(undecoded(tag, { noFrames: String(e?.message ?? e) }));
    }
  } else px = new Uint8Array(m.rgba);
  const t = performance.now();
  const r = phy.decode(px, w, h, null, isRgba)[0];
  // heap: the page watches this for a step, which is a heap growth and a stall the ms above already paid for.
  // quad: the lattice corners the finder settled on, in camera pixels. The page turns it into the symbol's size
  // and the pixels a cycle of the finest ring got, which is what decides how much of the frame comes back.
  // ai: ids and bytes: every block the frame verified and the judge did not call bad, repeats included (the page says
  // ai: which are new, since 2026-09-29), whatever they carry: whether they are a file's is the light's to say (its
  // ai: header, the fountain worker).
  const out = { type: "result", tag, found: r.found, seen: r.seen, bad: r.bad, test: r.test ? 1 : 0, judged: r.judged, heap: heapTop(), quad: r.found ? r.quad : null, built: r.built ? 1 : 0, ids: r.got.map((g) => g.id) };
  if (r.got.length) {
    const B = phy.usefulBytes, bytes = new Uint8Array(B * r.got.length);
    r.got.forEach((g, k) => bytes.set(g.bytes, k * B));
    out.bytes = bytes.buffer;
  }
  out.ms = performance.now() - t;
  if (vf) out.vf = vf;
  if (m.verify) out.probe = cornerProbe(px, w, h, isRgba ? 4 : 1);   // ai: the page cannot see a VideoFrame's pixels, so the crop check is answered from here
  // What the band said (src/fmt.h): the version, and the rate the sender means to paint at. Null on a frame that
  // read no word. The page keeps the last one, so this travels per frame and is averaged nowhere.
  // ai: The ring the word was read in rides with it (an index into sim/lizard_pick.mjs RINGS).
  if (r.fmt) out.fmt = { ...r.fmt, ring: r.ring };
  if (prof && phy.prof) out.prof = phy.prof();   // recv.html?prof: where the decode's time goes on this device
  // The grey range this frame actually arrived with, every thirtieth frame so it costs nothing. LIZARD's data is
  // in grey LEVELS, so what matters is not sharpness but how much of 0..255 the camera gave and whether it
  // clipped: a run at 41% of range with 13% of the picture pinned white read 17% of its blocks where one at 87%
  // read all of them. The middle half only, so the surround and the frame do not vote.
  if ((levelTick++ & 31) === 0) {
    const hist = new Uint32Array(256);
    const step = isRgba ? 4 : 1, x0 = w >> 2, x1 = (3 * w) >> 2, y0 = h >> 2, y1 = (3 * h) >> 2;
    let n = 0;
    for (let y = y0; y < y1; y += 2) for (let x = x0; x < x1; x += 2) { hist[px[(y * w + x) * step]]++; n++; }
    if (n) {
      const at = (p) => { let acc = 0; for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= n * p) return v; } return 255; };
      out.levels = { range: at(0.95) - at(0.05), mid: at(0.5), hi: hist[255] / n, lo: hist[0] / n };
    }
  }
  postMessage(out, out.bytes ? [out.bytes] : []);
}

onmessage = ({ data: m }) => {
  chain = chain.then(() => (m.type === "config" ? configure(m.config) : frame(m))).catch((e) => { m.frame?.close(); postMessage({ type: "error", tag: m.tag, message: String(e?.message ?? e) }); });
};
