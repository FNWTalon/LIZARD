// zstd, the transfer's compression (src/xfer.h: a chunk goes as one zstd frame where that is shorter than its bytes),
// through liblizard/zstd/shim.c, which pins the parameters so every sender writes the same frame. build/zstd.mjs comes
// from ./build.sh zstd. Handed to XferSender and XferReceiver (sim/xfer.mjs) as `Z`, the way Wirehair's classes are,
// so a module that only needs the id test loads neither.
import create from "../build/zstd.mjs";

let M;
export async function init() {
  if (!M) M = await create();
  return M;
}

// The frame of `bytes`, or null where the shim refuses (it does not, for a chunk the transfer allows).
export function compress(bytes) {
  const cap = M._lizard_zstd_bound(bytes.length), p = M._malloc(Math.max(1, bytes.length)), o = M._malloc(cap);
  M.HEAPU8.set(bytes, p);
  const n = M._lizard_zstd_compress(p, bytes.length, o, cap);
  const out = n < 0 ? null : M.HEAPU8.slice(o, o + n);
  M._free(p); M._free(o);
  return out;
}

// The `len` bytes a frame holds, or null where it is not a frame of exactly len bytes.
export function decompress(bytes, len) {
  const p = M._malloc(Math.max(1, bytes.length)), o = M._malloc(Math.max(1, len));
  M.HEAPU8.set(bytes, p);
  const rc = M._lizard_zstd_decompress(p, bytes.length, o, len);
  const out = rc ? null : M.HEAPU8.slice(o, o + len);
  M._free(p); M._free(o);
  return out;
}

// What the transfer takes: `Z` for sim/xfer.mjs.
export const Z = { compress, decompress };
