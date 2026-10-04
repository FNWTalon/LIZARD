// Wirehair, the project's fixed outer code, for end-to-end runs. build/wirehair.mjs comes from
// ./build.sh wirehair (liblizard/vendor/wirehair plus liblizard/wirehair/shim.cpp, compiled by path).
import create from "../build/wirehair.mjs";

let M;
export async function init() {
  if (!M) { M = await create(); if (M._lizard_wh_init() !== 0) throw new Error("wirehair init failed"); }
  return M;
}

export class Encoder {
  constructor(message, blockBytes) {
    const p = M._malloc(message.length), out = M._malloc(4);
    M.HEAPU8.set(message, p);
    this.seed = M._lizard_wh_encoder_create(p, message.length, blockBytes, out);
    M._free(p);
    if (this.seed < 0) throw new Error(`wirehair encoder: ${this.seed}`);
    this.h = M.getValue(out, "i32"); M._free(out);
    this.blockBytes = blockBytes; this.buf = M._malloc(blockBytes);
    this.blocks = Math.ceil(message.length / blockBytes);
  }
  // Every block is blockBytes long on the wire; the last source block is zero padded.
  block(id, out = new Uint8Array(this.blockBytes)) {
    const n = M._lizard_wh_encode(this.h, id, this.buf, this.blockBytes);
    if (n < 0) throw new Error(`wirehair encode: ${n}`);
    out.fill(0); out.set(M.HEAPU8.subarray(this.buf, this.buf + n));
    return out;
  }
  free() { M._lizard_wh_free(this.h); M._free(this.buf); }
}

export class Decoder {
  constructor(messageBytes, blockBytes, seed) {
    const out = M._malloc(4), rc = M._lizard_wh_decoder_create(messageBytes, blockBytes, seed, out);
    if (rc < 0) throw new Error(`wirehair decoder: ${rc}`);
    this.h = M.getValue(out, "i32"); M._free(out);
    Object.assign(this, { messageBytes, blockBytes, buf: M._malloc(blockBytes), lastId: Math.ceil(messageBytes / blockBytes) - 1 });
  }
  // true once enough blocks are in.
  add(id, bytes) {
    // The final source block travels padded; wirehair wants its true length.
    const n = id === this.lastId ? this.messageBytes - this.lastId * this.blockBytes : this.blockBytes;
    M.HEAPU8.set(bytes.subarray(0, n), this.buf);
    const rc = M._lizard_wh_decode(this.h, id, this.buf, n);
    if (rc < 0) throw new Error(`wirehair decode: ${rc}`);
    return rc === 1;
  }
  recover() {
    const p = M._malloc(this.messageBytes), n = M._lizard_wh_recover(this.h, p, this.messageBytes);
    if (n < 0) throw new Error(`wirehair recover: ${n}`);
    const out = M.HEAPU8.slice(p, p + n); M._free(p);
    return out;
  }
  free() { M._lizard_wh_free(this.h); M._free(this.buf); }
}

// Whole-file check for the rig: a length match alone would pass a wrong recovery.
export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) { c ^= bytes[i]; for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1)); }
  return ~c >>> 0;
}
