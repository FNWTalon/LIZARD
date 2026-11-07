// ai: The receiver's GPU decoder (recv.mjs Decoder GPU): one worker running the WebGPU host (../liblizard/gpu/host.mjs)
// ai: over the packed tree (../liblizard/gpu/webgpu.zst), the native receiver's way: frames staged on the ring as they
// ai: come, a batch launched as soon as a lane is free and a frame waits (at most the batcher's size), two in flight,
// ai: each with the last word read (the held version). It only decodes: the page dedupes and the fountain worker
// ai: makes the file, as for the C pool. Blocks are judged from the light (sim/phy.mjs blockJudge) as the C's are.
// ai:   page to worker: { type: "config", config } (as recv-worker.mjs); { type: "frame", tag, frame, x, y, w, h } (a
// ai:     VideoFrame, transferred, and its crop, copied to luma here: vframes.mjs lumaOf) or { type: "frame", tag, w, h,
// ai:     rgba } (the canvas grab).
// ai:   worker to page: { type: "ready", version, frames: true, take, adapter } (take: frames the page may have out at
// ai:     once, the ring's slots); a { type: "result", ... } a frame, recv-worker.mjs's fields; { type: "error", tag,
// ai:     message, fatal } (fatal: no GPU decoder here or the device lost; the page goes back to the C pool).
import { FrontHalf, deviceDescriptor, treeReader } from "../liblizard/gpu/host.mjs";
import { blockJudge } from "../liblizard/sim/phy.mjs";
import { init as initZstd, decompress } from "../liblizard/sim/zstd.mjs";
import { heapTop, init as initOb } from "../liblizard/sim/ob.mjs";
import { greyLevels, lumaOf } from "./vframes.mjs";

let fh = null, opening = null, dead = "";
let held = 0, levelTick = 0;
const judge = blockJudge();
const staged = [];   // ai: frames on the ring, not yet in a batch: { slot, tag, w, h, levels }
const waiting = [];  // ai: frames larger than the ring holds, as luma, until the lanes are planned again
let flight = Promise.resolve(), inFlight = 0, planning = false;

const fail = (message, tag) => { dead = message; postMessage({ type: "error", tag, message, fatal: true }); };

// ai: The device and the decoder, made once: the adapter's limits, the tree unpacked by the codec's zstd.
async function open() {
  if (!navigator.gpu) throw new Error("no WebGPU in this browser");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const desc = deviceDescriptor(adapter);
  if (desc.why) throw new Error(desc.why);
  const device = await adapter.requestDevice(desc);
  device.lost.then((i) => { if (i.reason !== "destroyed") fail(`the GPU device was lost: ${i.message}`); });
  const [z] = await Promise.all([fetch(new URL("../liblizard/gpu/webgpu.zst", import.meta.url)).then((r) => { if (!r.ok) throw new Error(`the WebGPU tree: ${r.status}`); return r.arrayBuffer(); }), initZstd(), initOb()]);   // ai: the codec's module: zstd for the tree, the test stream's generator for the judge
  const zb = new Uint8Array(z);
  // ai: the frame's content size (its header: single segment or not, the size field's width), which decompress asks
  const fhd = zb[4], fcs = fhd >> 6, single = (fhd >> 5) & 1, did = [0, 1, 2, 4][fhd & 3];
  const at = 5 + (single ? 0 : 1) + did, dv = new DataView(z);
  const len = fcs === 0 ? (single ? zb[at] : -1) : fcs === 1 ? dv.getUint16(at, true) + 256 : fcs === 2 ? dv.getUint32(at, true) : Number(dv.getBigUint64(at, true));
  const raw = len > 0 ? decompress(zb, len) : null;
  if (!raw) throw new Error("the WebGPU tree did not unpack");
  const made = await FrontHalf.create(device, treeReader(raw), { log: (m) => console.log(`gpu: ${m}`) });
  return { fh: made, adapter: [adapter.info?.vendor, adapter.info?.architecture, adapter.info?.description].filter(Boolean).join(" ") };
}

async function configure(c) {
  if (dead) return fail(dead);
  try {
    opening ??= open();
    const o = await opening;
    fh = o.fh;
    postMessage({ type: "ready", version: c.version, frames: true, take: fh.ring?.R ?? fh.C.B_CEIL + fh.C.RING_SLACK, adapter: o.adapter });
  } catch (e) { fail(String(e?.message ?? e)); }
}

const undecoded = (tag, more = {}) => ({ type: "result", tag, found: 0, seen: 0, bad: 0, ms: 0, ids: [], ...more });

// ai: A frame onto the ring (the lanes planned at its size first, and again for a larger one once the batches in
// ai: flight are in), then a launch if a lane is free.
async function frame(m) {
  if (!fh || dead) { m.frame?.close(); return postMessage(undecoded(m.tag)); }
  let px, vf = null;
  const { w, h, tag } = m;
  if (m.frame) {
    try { ({ luma: px, format: vf } = await lumaOf(m.frame, m.x, m.y, w, h)); }
    catch (e) { return postMessage(undecoded(tag, { noFrames: String(e?.message ?? e) })); }
  } else {
    const rgba = new Uint8Array(m.rgba);
    px = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < px.length; i++, j += 4) px[i] = (77 * rgba[j] + 150 * rgba[j + 1] + 29 * rgba[j + 2] + 128) >> 8;   // ai: the rig's formula (src/acquire.c), as lumaOf's
  }
  const levels = (levelTick++ & 31) === 0 ? greyLevels(px, w, h, 1) : null;
  const one = { tag, w, h, px, vf, levels };
  if (!fh.ring || w > fh.ring.W || h > fh.ring.H || planning || waiting.length) { waiting.push(one); launch(); return replan(); }
  stage(one);
  launch();
}

function stage(one) {
  const slot = fh.enqueueLuma(one.px, one.w, one.h, one.w, one.tag);
  if (!slot) return postMessage(undecoded(one.tag));   // ai: the ring full (the page has more out than it was told to)
  one.px = null;
  staged.push({ ...one, slot });
}

// ai: The lanes planned for the largest frame waiting, once nothing is in flight or staged on the old ring.
async function replan() {
  if (planning || inFlight || staged.length) return;
  planning = true;
  try {
    let W = 0, H = 0;
    for (const f of waiting) { W = Math.max(W, f.w); H = Math.max(H, f.h); }
    await fh.plan(W, H);
  } catch (e) { planning = false; return fail(`the GPU decoder could not plan ${e?.message ?? e}`); }
  planning = false;
  // ai: a frame that came while planning and is larger still waits for the next plan
  const again = [];
  for (const f of waiting.splice(0)) if (f.w <= fh.ring.W && f.h <= fh.ring.H) stage(f); else again.push(f);
  waiting.push(...again);
  launch();
  if (waiting.length) replan();
}

function launch() {
  while (staged.length && fh.freeLane() && inFlight < fh.inflightMax()) {
    const batch = staged.splice(0, fh.size());
    let f;
    try { f = fh.run(batch.map((b) => b.slot), held); }
    catch (e) { for (const b of batch) postMessage(undecoded(b.tag)); fail(`a batch failed: ${e?.message ?? e}`); return; }
    inFlight++;
    const done = fh.finish(f);
    // ai: read back in launch order, so the held word moves as the frames came
    flight = flight.then(() => done).then((out) => answer(out, batch), (e) => { for (const b of batch) postMessage(undecoded(b.tag)); fail(`a batch's readback failed: ${e?.message ?? e}`); })
      .finally(() => { inFlight--; launch(); if (waiting.length) replan(); });
  }
}

// ai: Each frame's answer as recv-worker.mjs gives it: the verified blocks judged against the test stream, the word,
// ai: the corners the finder settled on (the tracker's), the decode's share of its batch's GPU time.
function answer(out, batch) {
  const ms = out.deviceMs / Math.max(1, out.carried), P = fh.C.PAYLOAD, B = P - 4;
  out.frames.forEach((fo, i) => {
    const b = batch[i];
    if (fo.hasWord) held = fo.word.version;
    const n = fo.records.length, blocks = new Uint8Array(n * P);
    fo.records.forEach((r, k) => blocks.set(r.payload, k * P));
    const j = judge.frame(new Uint8Array(n).fill(1), blocks);
    const found = fo.finder.found === 1;
    const msg = { type: "result", tag: b.tag, found: found ? 1 : 0, seen: j.seen, bad: j.bad, test: j.test ? 1 : 0, judged: j.judged, heap: heapTop(), quad: found ? fo.finder.corners : null, ids: j.got.map((g) => g.id), ms, gpu: 1 };
    if (j.got.length) { const bytes = new Uint8Array(B * j.got.length); j.got.forEach((g, k) => bytes.set(g.bytes, k * B)); msg.bytes = bytes.buffer; }
    if (fo.hasWord && fo.ring >= 0) msg.fmt = { version: fo.word.version, fps: fo.word.fps, ring: fo.ring };
    if (b.vf) msg.vf = b.vf;
    if (b.levels) msg.levels = b.levels;
    postMessage(msg, msg.bytes ? [msg.bytes] : []);
  });
}

let chain = Promise.resolve();
onmessage = ({ data: m }) => {
  chain = chain.then(() => (m.type === "config" ? configure(m.config) : frame(m))).catch((e) => { m.frame?.close(); postMessage({ type: "error", tag: m.tag, message: String(e?.message ?? e) }); });
};
