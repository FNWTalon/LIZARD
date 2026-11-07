// ai: The sender's transfer made by a pool of workers (2026-11-06; the native sender's prepare across cores,
// ai: liblizard/core/tx/xfer_tx.cpp). No chunk reads another, so each worker (chunk-worker.mjs) takes the next chunk as it
// ai: comes free and makes it whole (its zstd frame, its fountain, its chaining value); Wirehair runs in parallel too,
// ai: as each worker keeps the fountains it made (a wasm heap's state cannot move to another), and so the fountains
// ai: spread over the workers' heaps (one heap holds 2 GiB at most). A frame's blocks are fetched from the workers that
// ai: hold them before it is painted (send-worker.mjs). The ids and bytes are the serial sender's (sim/xfer.mjs).
import { XferSender, DEFAULT_LOG2, PAYLOAD, SYMBOL_BITS, CONTROL } from "../liblizard/sim/xfer.mjs";

// ai: a worker's answers taken in the order they were asked (it handles one message at a time)
function link(w) {
  const waiting = [];
  w.onmessage = ({ data }) => { const p = waiting.shift(); if (data?.error) p.reject(new Error(data.error)); else p.resolve(data); };
  w.onerror = (e) => { e.preventDefault(); for (const p of waiting.splice(0)) p.reject(new Error(e.message || "a chunk worker failed")); };
  return (msg, transfer = []) => new Promise((resolve, reject) => { waiting.push({ resolve, reject }); w.postMessage(msg, transfer); });
}

// ai: The file's transfer (XferSender's options in opts), made by up to `workers` workers, one a chunk at most; a file of
// ai: one chunk, or one worker, made here as before.
export async function makeXfer(bytes, opts, { M, Encoder, Z, workers = navigator.hardwareConcurrency || 1 }) {
  const chunkLog2 = opts.chunkLog2 ?? DEFAULT_LOG2, size = 2 ** chunkLog2, C = Math.ceil(bytes.length / size);
  const P = Math.min(C, workers);
  if (P < 2) return new XferSender(bytes, { ...opts, M, Encoder, Z });
  const ws = Array.from({ length: P }, () => new Worker(new URL("./chunk-worker.mjs", import.meta.url), { type: "module" }));
  const calls = ws.map(link), owner = new Int32Array(C), chunks = new Array(C), free = () => ws.forEach((w) => w.terminate());
  try {
    let next = 0;
    await Promise.all(calls.map(async (call, i) => {
      for (let c; (c = next++) < C;) {
        const plain = bytes.slice(c * size, Math.min(bytes.length, (c + 1) * size));
        chunks[c] = await call({ type: "make", c, chunkLog2, zstd: !!Z, buf: plain.buffer }, [plain.buffer]);
        owner[c] = i;
      }
    }));
    const remote = {
      // ai: the blocks of the ids whose chunks the workers hold (a fountain's), each worker asked once for its own
      async fetch(ids) {
        const per = calls.map(() => []), got = new Map();
        for (const id of ids) { const c = id >>> SYMBOL_BITS; if (c !== CONTROL && chunks[c].K > 1) per[owner[c]].push(id); }
        await Promise.all(per.map(async (list, i) => {
          if (!list.length) return;
          const b = new Uint8Array((await calls[i]({ type: "blocks", ids: Uint32Array.from(list) })).buf);
          list.forEach((id, k) => got.set(id, b.subarray(k * PAYLOAD, (k + 1) * PAYLOAD)));
        }));
        return got;
      },
      free,
    };
    return new XferSender(null, { ...opts, M, Encoder, Z, chunks, length: bytes.length, remote });
  } catch (e) { free(); throw e; }
}
