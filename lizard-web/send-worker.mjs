// The sender's encoder, off the page's thread (lizard-web/send.mjs), for a Lizard symbol. It builds each frame's
// blocks (the test stream, or the fountain for a file), encodes them and writes the finished RGBA pixels in wasm
// (src/focus.c focus_paint_rgba), a few frames ahead of the page, so the page only puts pixels on its canvas. At
// LIZARD-1024 that is 13 ms here and a few on the page, where the page used to spend 45 ms a frame on both.
//
// Messages run one at a time in arrival order (the chain below), so a frame asked for after a re-pick is encoded with
// the new code and one asked for before it with the old: the page's block ids stay right across the change.
//
// A file goes as a chunked transfer (sim/xfer.mjs): its header and manifest in the light, a fountain per chunk, and the
// block ids chosen here by the transfer's schedule, so the page's baseId only counts blocks.
// ai: codes: symbols a frame, 1 or 2 (the page's "codes"). Two are painted side by side in one buffer, a gap of
// ai: GAP_MODULES modules between them (2026-09-30: widened by about 2 modules, then 2 more: 4; then 12, as the 8
// ai: modules of white between the rims could not be seen; packed to their quiet zones from 2026-09-28, since with two
// ai: codes the whole performance rests on packing), the first with the frame's first blocksPerFrame ids and the second
// ai: with the next, one format and one word for both.
// ai: The gap is the same modules in the page's GPU encoder (gpu/encoder.mjs GAP_MODULES): change both.
// ai: Under the page's GPU encoder (gpu/encoder.mjs, since 2026-09-29) frames are asked for a batch at a time
// ai: ("frames": count frames from seq) and come back as their blocks alone, one buffer, frame after frame, each codes x
// ai: blocksPerFrame x 473 bytes; the page encodes them.
import { makePhy } from "../liblizard/sim/phy.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initFountain, Encoder } from "../liblizard/sim/fountain.mjs";
import { XferSender, isControlId } from "../liblizard/sim/xfer.mjs";

let phy = null, xs = null, fps = 0, codes = 1, chain = Promise.resolve();
const source = () => (id, out) => xs.block(id, out);
// ai: the gap between two codes, in modules (the page's slider, start and respec's gap; GAP_MODULES unless given), and
// ai: in the paint's pixels for this codec (phy.cell samples a module)
export const GAP_MODULES = 12;
let gapModules = GAP_MODULES;
const gapOf = (p) => gapModules * (p.cell || 0);

async function handle(m) {
  if (m.type === "start") {
    phy?.free(); xs?.free(); phy = null; xs = null;
    phy = await makePhy(m.spec);
    if (m.bytes) {
      await initFountain();
      xs = new XferSender(new Uint8Array(m.bytes), { ...m.xfer, M: await initOb(), Encoder });
      phy.setSource(source());
    }
    fps = m.fps; phy.setFps?.(fps); codes = m.codes ?? 1; gapModules = m.gap ?? GAP_MODULES;
    postMessage({ type: "ready", gen: m.gen, label: phy.label, usefulBytes: phy.usefulBytes, blocksPerFrame: phy.blocksPerFrame, xfer: xs?.info() });
  } else if (m.type === "respec") {
    // A re-pick (the room changed): a new code for the same transfer. Refused if the block size would move, which
    // would break the fountain's stream, exactly as the page used to refuse it.
    const next = await makePhy(m.spec);
    if (next.usefulBytes !== phy.usefulBytes) { next.free(); postMessage({ type: "respec", gen: m.gen, ok: false }); return; }
    phy.free(); phy = next; phy.setFps?.(fps); codes = m.codes ?? codes; gapModules = m.gap ?? gapModules;
    if (xs) phy.setSource(source());
    postMessage({ type: "respec", gen: m.gen, ok: true, label: phy.label, blocksPerFrame: phy.blocksPerFrame });
  } else if (m.type === "fps") {
    fps = m.fps; phy?.setFps?.(fps);
  } else if (m.type === "stop") {
    phy?.free(); xs?.free(); phy = null; xs = null;
  } else if (m.type === "frame") {
    if (!phy) return;
    const t = performance.now(), V = phy.blocksPerFrame, ids = xs ? xs.frameIds(V * codes) : null;
    // ai: Each symbol's pixels are a view into the wasm heap, good until the next paint, so each is copied row by row
    // ai: into its place (side k of codes) before the next is painted. The page hands its last buffer back to be filled.
    // ai: Two codes sit a gap of GAP_MODULES apart (since 2026-09-30; packed to their quiet zones before), the gap the
    // ai: colour of the margin beside it; the frame is
    // ai: codes w + (codes - 1) gap px wide and the message says the gap.
    let out = null, w = 0, gap = 0, fw = 0;
    for (let k = 0; k < codes; k++) {
      const part = phy.frameRGBA(m.seq, ids ? ids.slice(k * V, (k + 1) * V) : m.baseId + k * V), row = 4 * part.w;
      if (!out) {
        w = part.w; gap = codes > 1 ? gapOf(phy) : 0; fw = codes * w + (codes - 1) * gap;
        const bytes = 4 * fw * w;
        out = m.buf && m.buf.byteLength === bytes ? new Uint8Array(m.buf) : new Uint8Array(bytes);
      }
      for (let y = 0; y < w; y++) {
        const at = 4 * (y * fw + k * (w + gap));
        out.set(part.rgba.subarray(y * row, (y + 1) * row), at);
        if (k + 1 < codes) for (let x = 0; x < gap; x++) out.set(part.rgba.subarray((y + 1) * row - 4, (y + 1) * row), at + row + 4 * x);
      }
    }
    // ai: data: the frame's data blocks, its header and manifest blocks not counted; the page counts laps with it
    // ai: (xfer.lap data blocks a lap). Every block of the test stream is data.
    let data = V * codes;
    if (ids) { data = 0; for (const id of ids) if (!isControlId(id)) data++; }
    postMessage({ type: "frame", gen: m.gen, seq: m.seq, w, codes, gap, buf: out.buffer, ms: performance.now() - t, data }, [out.buffer]);
  } else if (m.type === "frames") {
    if (!phy) return;
    const t = performance.now(), V = phy.blocksPerFrame, B = phy.usefulBytes + 4, per = codes * V * B, bytes = m.count * per;
    const out = m.buf && m.buf.byteLength === bytes ? new Uint8Array(m.buf) : new Uint8Array(bytes), data = [];
    for (let f = 0; f < m.count; f++) {
      const ids = xs ? xs.frameIds(V * codes) : null, base = m.baseId + f * V * codes;
      for (let k = 0; k < codes; k++) out.set(phy.frameBlocks(m.seq + f, ids ? ids.slice(k * V, (k + 1) * V) : base + k * V), f * per + k * V * B);
      data.push(ids ? ids.filter((id) => !isControlId(id)).length : V * codes);
    }
    postMessage({ type: "frames", gen: m.gen, seq: m.seq, count: m.count, codes, buf: out.buffer, ms: performance.now() - t, data }, [out.buffer]);
  }
}

onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ type: "error", gen: m.gen, message: String(e?.message ?? e) }));
};
