// ai: GpuQueue (gpuqueue.mjs) on a fake FrontHalf built to the queue's contract (gpu/decoder.mjs: enqueue, release,
// ai: run on slots, ready on a frame's shape, needsPlan, size, auto, lanes, ring, inflight, device.lost), no GPU: items
// ai: pushed on a 60 fps clock, a batch's readback settling after a set delay. Asserts the hold before the ring, full
// ai: batches while arrivals keep up, a partial batch at the bound once they stop, at most fh.inflight computing at
// ai: once, the oldest staged dropped on a full ring, a fixed B launching full or on flush(), every run carrying
// ai: frames, every item answered exactly once with its source closed once, and stop() answering everything and
// ai: submitting nothing more. node lizard-web/check_gpuqueue.mjs, about 3 s; exit 1 on a failed line.
import { GpuQueue, HOLD_MAX } from "./gpuqueue.mjs";

const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) fails++; };

// ai: ms: a batch's compute; planMs: ready()'s; R: the ring's slots. Counts batches computing at once (most) and every
// ai: run (runs: its slot count), as the decoder refuses a batch of no slots.
class FakeFH {
  constructor({ B = 4, auto = true, R = 8, ms = 20, planMs = 40 }) {
    Object.assign(this, { B, auto, R, ms, planMs, size: auto ? 0 : B, ring: null, inflight: 2, lanes: [{ busy: false }, { busy: false }], next: 1, computing: 0, most: 0, batches: [], launchedAt: [], readyOn: [], stagedMax: 0, runs: [], lost: null });
    this.device = { lost: new Promise(() => {}), destroyed: 0, destroy: () => { this.device.destroyed++; } };
  }
  needsPlan() { return !this.ring; }
  async ready(frames) { this.readyOn.push(frames); await sleep(this.planMs); this.ring = { R: this.R, tex: null, staged: 0 }; this.size = this.B; }
  enqueue(f) {
    if (!this.ring) throw new Error("no ring");
    if (this.ring.staged >= this.R) return null;
    this.stagedMax = Math.max(this.stagedMax, ++this.ring.staged);
    return { id: this.next++, w: f.w, h: f.h, at: performance.now(), f0: !!f.source };
  }
  release() { this.ring.staged--; }
  run(slots, maps, opts = {}) {
    this.runs.push({ n: slots.length, opts });
    if (!slots.length) throw new Error("a batch of no slots");
    if (slots.length > this.B) throw new Error(`${slots.length} frames in a batch of at most ${this.B}`);
    const lane = this.lanes.find((l) => !l.busy);
    if (!lane) throw new Error("no free lane");
    this.ring.staged -= slots.length; this.batches.push(slots.length); this.launchedAt.push(performance.now());
    lane.busy = true; this.most = Math.max(this.most, ++this.computing);
    return sleep(this.ms).then(() => {
      this.computing--; lane.busy = false;
      return { frames: slots.map(() => ({})), deviceMs: this.ms, carried: slots.length };
    });
  }
}

// ai: A queue over a fake; every item carries luma and a source that counts its closes (the F0 path's).
// ai: doneTags: each done call's tags, in order.
function rig(opts) {
  const fh = new FakeFH(opts), items = [], dropped = [], doneTags = [];
  const answer = (list) => { for (const it of list) it.answered++; };
  const q = new GpuQueue({ fh, nmax: 1024 }, { reopen: async () => { throw new Error("no reopen here"); }, done: (list) => { answer(list); doneTags.push(list.map((it) => it.tag)); },
    drop: (list) => { answer(list); dropped.push(...list.map((it) => it.tag)); }, lost: (why) => check(false, `the queue gave up: ${why}`) });
  const push = () => { const it = { tag: items.length, w: 64, h: 64, x: 0, y: 0, luma: new Uint8Array(64 * 64), answered: 0, closed: 0 }; it.source = { close: () => it.closed++ }; items.push(it); q.push(it); return it; };
  const feed = (ms) => new Promise((ok) => { const t = setInterval(push, 1000 / 60); setTimeout(() => { clearInterval(t); ok(); }, ms); });
  return { fh, q, items, dropped, doneTags, push, feed };
}
const rigs = [];

// ai: 1: six frames before the plan lands: HOLD_MAX held, the oldest beyond dropped, the held enqueued once the ring is there.
{
  const r = rig({ B: 4 }); rigs.push(r);
  for (let i = 0; i < 6; i++) r.push();
  check(r.q.held === HOLD_MAX && r.dropped.join() === "0,1" && r.q.staged === 0, `before the ring: ${r.q.held} of 6 held, dropped ${r.dropped.join(",")} (the oldest two)`);
  await sleep(70);
  check(r.fh.readyOn.length === 1 && r.fh.readyOn[0][0].w === 64 && r.fh.readyOn[0][0].h === 64 && r.q.held === 0 && r.fh.batches[0] === 4, `ready() once on the held frame's shape; the held enqueued after it: first batch of ${r.fh.batches[0]}`);
  // ai: 2: a 60 fps feed, batches of 20 ms: every batch carries the target.
  const b0 = r.fh.batches.length;
  await r.feed(500);
  const bs = r.fh.batches.slice(b0);
  check(bs.length >= 5 && bs.every((b) => b === 4), `${bs.length} batches while arrivals keep up, each of ${[...new Set(bs)].join("/")} (target 4)`);
  // ai: 3: the feed's leftovers drained, two frames arrive and no more: a partial batch once the oldest has waited the bound.
  while (r.q.staged || r.q.computing.size) await sleep(10);
  const t0 = performance.now(), n0 = r.fh.batches.length;
  r.push(); r.push();
  const bound = r.q.bound;
  await sleep(bound + 100);
  const dt = r.fh.launchedAt[n0] - t0;
  check(r.fh.batches.length === n0 + 1 && r.fh.batches[n0] === 2 && dt >= bound - 2 && dt < bound + 60, `arrivals stopped: a batch of ${r.fh.batches[n0]} after ${dt.toFixed(0)} ms (bound ${bound.toFixed(0)})`);
  // ai: nothing more arrives: no run follows it, and stop() submits none.
  const runs = r.fh.runs.length;
  await sleep(bound + 100);
  await r.q.stop();
  check(r.fh.runs.length === runs, `then idle: no run after the partial batch, none at stop() (${r.fh.runs.length} in all)`);
}
// ai: 5: batches of 200 ms under a 60 fps feed on a ring of 5: the ring fills, the oldest staged is dropped, the next
// ai: batch carries newer frames than any dropped.
{
  const r = rig({ B: 4, R: 5, ms: 200 }); rigs.push(r);
  r.push(); await sleep(70);
  await r.feed(300);
  await sleep(450);   // ai: the readbacks of the batches in flight and of the one launched after them
  const later = r.doneTags.at(-1) ?? [];
  check(r.dropped.length > 0 && r.fh.stagedMax <= 5 && later.length === 4 && Math.min(...later) > Math.max(...r.dropped), `full ring: ${r.dropped.length} dropped (the oldest), the ring at most ${r.fh.stagedMax} of 5, the newest batch's tags ${later.join(",")} past the dropped`);
  await r.q.stop();
}
// ai: 8: a fixed B has no bound: two frames launch, one waits until flush().
{
  const r = rig({ auto: false, B: 2 }); rigs.push(r);
  r.push(); await sleep(70); r.push();
  const two = r.fh.batches[0] === 2;
  r.push(); await sleep(100);
  const waited = r.fh.batches.length === 1 && r.q.staged === 1;
  r.q.flush(); await sleep(10);
  check(two && waited && r.fh.batches[1] === 1, `fixed B 2: a batch of ${r.fh.batches[0]}, then one frame waited ${waited ? "" : "not "}for flush(): a batch of ${r.fh.batches[1]}`);
  await sleep(50);
  await r.q.stop();
}
// ai: 7: stop() with a batch computing, frames staged and frames pushed after: all answered, the computing batch
// ai: delivered, nothing submitted after it, the device destroyed once.
{
  const r = rig({ B: 4, ms: 100 }); rigs.push(r);
  r.push(); await sleep(70);
  for (let i = 0; i < 6; i++) r.push();
  const runs = r.fh.runs.length;
  const p = r.q.stop();
  r.push();
  await p;
  check(r.items.every((it) => it.answered === 1) && r.fh.device.destroyed === 1 && r.doneTags.flat().length === 4 && r.fh.runs.length === runs, `stop(): ${r.items.length} items answered (${r.doneTags.flat().length} decoded, ${r.dropped.length} dropped), no run after it, the device destroyed ${r.fh.device.destroyed} time`);
}
// ai: 4, 6 and 10, over every rig.
check(rigs.every((r) => r.fh.most <= r.fh.inflight), `never more than fh.inflight batches computing at once (most ${rigs.map((r) => r.fh.most).join(",")} of ${rigs[0].fh.inflight})`);
check(rigs.every((r) => r.fh.runs.every((x) => x.n > 0 && !x.opts.cancelOnly)), `every run carried frames (${rigs.reduce((a, r) => a + r.fh.runs.length, 0)} runs)`);
const all = rigs.flatMap((r) => r.items);
check(all.every((it) => it.answered === 1 && it.closed === 1), `every item answered once and its source closed once (${all.length} items, ${all.filter((it) => it.answered !== 1).length} off)`);
const decoded = rigs.flatMap((r) => r.doneTags.flat().map((t) => `${rigs.indexOf(r)}:${t}`));
check(new Set(decoded).size === decoded.length && rigs.every((r) => r.doneTags.every((tags) => tags.length > 0)), `every decoded item in exactly one done call, and no done call empty (${decoded.length} items)`);
process.exit(fails ? 1 : 0);
