// ai: A worker of the sender's pool (xfer-pool.mjs, 2026-11-06): makes each chunk it is handed (sim/xfer.mjs makeChunk:
// ai: its zstd frame where shorter, its fountain, its chaining value), keeps the fountain and answers the blocks a frame
// ai: asks of it. One message at a time, in order (the pool waits on its answers in the order it asked).
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initFountain, Encoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
import { makeChunk, PAYLOAD, SYMBOL_BITS, SYMBOL_MASK } from "../liblizard/sim/xfer.mjs";

const enc = new Map();
let M = null, chain = Promise.resolve();

async function handle(m) {
  if (m.type === "make") {
    if (!M) [M] = await Promise.all([initOb(), initFountain(), initZstd()]);
    const k = makeChunk(new Uint8Array(m.buf), m.c, m.chunkLog2, true, { M, Encoder, Z: m.zstd ? Z : null });
    if (k.enc) enc.set(m.c, k.enc);
    postMessage({ sent: k.sent, K: k.K, seed: k.seed, one: k.one, cv: k.cv });
  } else if (m.type === "blocks") {
    const out = new Uint8Array(m.ids.length * PAYLOAD);
    m.ids.forEach((id, i) => enc.get(id >>> SYMBOL_BITS).block(id & SYMBOL_MASK, out.subarray(i * PAYLOAD, (i + 1) * PAYLOAD)));
    postMessage({ buf: out.buffer }, [out.buffer]);
  }
}

onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ error: String(e?.message ?? e) }));
};
