// ai: A worker of the sender's pool (xfer-pool.mjs, 2026-11-06): hashes the chunks it is handed (their chaining values,
// ai: the pool's first pass), makes each chunk it is handed after (sim/xfer.mjs makeChunk: its zstd frame where shorter,
// ai: its fountain), keeps the fountain and answers the blocks a frame asks of it; a trial chunk for the pool's timing of
// ai: the cores is made and let go. One message at a time, in order (the pool waits on its answers in the order it asked).
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initFountain, Encoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
import { makeChunk, chunkCv, PAYLOAD, SYMBOL_BITS, SYMBOL_MASK } from "../liblizard/sim/xfer.mjs";

const enc = new Map();
let M = null, chain = Promise.resolve();

async function handle(m) {
  if (!M && m.type !== "blocks") [M] = await Promise.all([initOb(), initFountain(), initZstd()]);
  if (m.type === "cv") postMessage({ cv: chunkCv(M, new Uint8Array(m.buf), m.c, m.chunkLog2) });
  else if (m.type === "make") {
    const k = makeChunk(new Uint8Array(m.buf), m.c, m.chunkLog2, false, { M, Encoder, Z: m.zstd ? Z : null });
    if (k.enc) enc.set(m.c, k.enc);
    postMessage({ sent: k.sent, K: k.K, seed: k.seed, one: k.one });
  } else if (m.type === "trial") {
    // ai: the pool's timing of the cores (xfer-pool.mjs): a chunk made and let go
    makeChunk(new Uint8Array(m.buf), 0, m.chunkLog2, true, { M, Encoder, Z: m.zstd ? Z : null }).enc?.free();
    postMessage({});
  } else if (m.type === "blocks") {
    const out = new Uint8Array(m.ids.length * PAYLOAD);
    m.ids.forEach((id, i) => enc.get(id >>> SYMBOL_BITS).block(id & SYMBOL_MASK, out.subarray(i * PAYLOAD, (i + 1) * PAYLOAD)));
    postMessage({ buf: out.buffer }, [out.buffer]);
  }
}

onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ error: String(e?.message ?? e) }));
};
