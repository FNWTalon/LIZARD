// ai: The sender's transfer made by a pool of workers (2026-11-06; the native sender's prepare across cores,
// ai: liblizard/core/tx/xfer_tx.cpp). No chunk reads another, so each worker (chunk-worker.mjs) takes the next chunk as it
// ai: comes free and makes it whole (its zstd frame, its fountain, its chaining value); Wirehair runs in parallel too,
// ai: as each worker keeps the fountains it made (a wasm heap's state cannot move to another), and so the fountains
// ai: spread over the workers' heaps (one heap holds 2 GiB at most). A frame's blocks are fetched from the workers that
// ai: hold them before it is painted (send-worker.mjs). The ids and bytes are the serial sender's (sim/xfer.mjs).
// ai: Streaming (2026-11-06, the native sender's): the workers hash every chunk first (the chaining values and the root,
// ai: all the header needs) and makeXfer returns; the chunks are made behind it while frames go, each handed to the
// ai: sender as it comes (XferSender.add), remote.made settling with the last.
import { XferSender, DEFAULT_LOG2, PAYLOAD, SYMBOL_BITS, CONTROL } from "../liblizard/sim/xfer.mjs";

// ai: a worker's answers taken in the order they were asked (it handles one message at a time); end() refuses what is
// ai: still asked of a worker let go
function link(w) {
  const waiting = [];
  const fail = (why) => { for (const p of waiting.splice(0)) p.reject(new Error(why)); };
  w.onmessage = ({ data }) => { const p = waiting.shift(); if (data?.error) p.reject(new Error(data.error)); else p.resolve(data); };
  w.onerror = (e) => { e.preventDefault(); fail(e.message || "a chunk worker failed"); };
  const call = (msg, transfer = []) => new Promise((resolve, reject) => { waiting.push({ resolve, reject }); w.postMessage(msg, transfer); });
  call.end = () => fail("let go");
  return call;
}

// ai: The workers: one a core, not a hardware thread (the native sender's rule, liblizard/core/tx/xfer_tx.cpp workers()),
// ai: found by timing (2026-11-06). A page is told only hardwareConcurrency, its hardware threads, and nothing says
// ai: whether two of them share a core; so the file's first MiB is made as a chunk by half the threads' workers at once,
// ai: then by all of them at once, and all are kept only where doubling them made at least 1.5 times as much: a
// ai: second thread on a core gives about 1 to 1.3 (on a 16-core 32-thread x86 in Chrome, 32 workers made 256 MiB no
// ai: faster than 16, at twice the memory, about 30 MiB a worker), a core of its own about 2. A round first warms every
// ai: worker (its codec loaded, its heap grown). A MiB, not a whole chunk: still past a core's cache, at a quarter of
// ai: the cost (whole chunks took 1.16 s for 256 MiB that then took 0.57 s). Found once a page (the send worker's
// ai: module lives as long), and only where it decides anything: more chunks than half the threads.
let found = null;
async function timeCores(calls, sample, chunkLog2, zstd) {
  const n = calls.length, half = n >> 1;
  const round = async (k) => {
    const t = performance.now();
    await Promise.all(calls.slice(0, k).map((call) => call({ type: "trial", buf: sample, chunkLog2, zstd })));
    return performance.now() - t;
  };
  const t0 = performance.now();
  await round(n);
  const a = half / await round(half), b = n / await round(n);
  return { cores: b >= 1.5 * a ? n : half, half: a, all: b, ms: Math.round(performance.now() - t0) };
}

// ai: The file's transfer (XferSender's options in opts), made by up to `workers` workers (the cores found unless given),
// ai: one a chunk at most; a file of one chunk, or one worker, made here as before.
export async function makeXfer(bytes, opts, { M, Encoder, Z, workers = null }) {
  const chunkLog2 = opts.chunkLog2 ?? DEFAULT_LOG2, size = 2 ** chunkLog2, C = Math.ceil(bytes.length / size);
  const n = navigator.hardwareConcurrency || 1, timed = workers == null && found == null && n >= 4 && C > n >> 1;
  let P = Math.min(C, workers ?? found?.cores ?? n);
  if (P < 2) return new XferSender(bytes, { ...opts, M, Encoder, Z });
  const ws = Array.from({ length: timed ? n : P }, () => new Worker(new URL("./chunk-worker.mjs", import.meta.url), { type: "module" }));
  const calls = ws.map(link), owner = new Int32Array(C), meta = new Array(C);
  const free = () => { ws.forEach((w) => w.terminate()); calls.forEach((call) => call.end()); };
  try {
    const part = (c) => bytes.slice(c * size, Math.min(bytes.length, (c + 1) * size));
    // ai: each worker the next chunk as it comes free (the work of a pass), the pass done when every one is
    const pass = (job) => { let next = 0; return Promise.all(calls.map(async (call, i) => { for (let c; (c = next++) < C;) await job(call, i, c); })); };
    const cvs = new Uint8Array(32 * C);
    await pass(async (call, i, c) => { const plain = part(c); cvs.set((await call({ type: "cv", c, chunkLog2, buf: plain.buffer }, [plain.buffer])).cv, 32 * c); });
    let xs = null;
    const remote = {
      // ai: the blocks of the ids whose chunks the workers hold (a fountain's), each worker asked once for its own
      async fetch(ids) {
        const per = calls.map(() => []), got = new Map();
        for (const id of ids) { const c = id >>> SYMBOL_BITS; if (c !== CONTROL && meta[c].K > 1) per[owner[c]].push(id); }
        await Promise.all(per.map(async (list, i) => {
          if (!list.length) return;
          const b = new Uint8Array((await calls[i]({ type: "blocks", ids: Uint32Array.from(list) })).buf);
          list.forEach((id, k) => got.set(id, b.subarray(k * PAYLOAD, (k + 1) * PAYLOAD)));
        }));
        return got;
      },
      free,
      workers: P,
      timing: timed ? found : null,   // ai: the cores' timing where this transfer made it (chunks a second at half, at all)
      made: null,                     // ai: settles once every chunk is in xs (or a worker failed)
    };
    xs = new XferSender(null, { ...opts, M, Encoder, Z, cvs, length: bytes.length, remote });
    // ai: the cores timed after the header (every thread's worker hashed: a short pass), the extra workers let go before
    // ai: any chunk is made, so the first frame waits on the hash alone
    const cores = async () => {
      if (!timed) return;
      found = await timeCores(calls, bytes.slice(0, Math.min(size, 2 ** 20)), chunkLog2, !!Z);
      P = Math.min(C, found.cores);
      ws.splice(P).forEach((w) => w.terminate());
      calls.splice(P);
      Object.assign(remote, { workers: P, timing: found });
    };
    remote.made = cores().then(() => pass(async (call, i, c) => {
      const plain = part(c), k = await call({ type: "make", c, chunkLog2, zstd: !!Z, buf: plain.buffer }, [plain.buffer]);
      meta[c] = k; owner[c] = i;
      xs.add(c, k);
    }));
    remote.made.catch(() => {});   // ai: the caller's to read; a transfer let go before it is whole ends it so
    return xs;
  } catch (e) { free(); throw e; }
}
