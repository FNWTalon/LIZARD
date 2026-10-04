// A tar of a recorded run, built in the page so the frames never go over the network one at a time.
//
// The rig used to upload each frame as it was decoded: 1.17 MB a frame at the 1080 crop, 50 a second offered,
// which is 58 MB/s over the phone's wifi and it simply does not fit. tar because it needs no compression, no
// library and no CRC table, writes as a list of buffers a Blob can hold without ever concatenating them, and
// lands on the other machine with `tar -xf` and nothing else.
//
// Pure, so build/tar_check.mjs can exercise it in node against real tar.

const BLOCK = 512;
const pad = (n) => (BLOCK - (n % BLOCK)) % BLOCK;

// One ustar header. Names are kept short and plain (NNNN.gray, meta.json), so none of the prefix field,
// long-name or pax extensions are needed; a name that would not fit is a bug in the caller, not a case to handle.
function header(name, size, mtime) {
  const h = new Uint8Array(BLOCK), enc = new TextEncoder();
  const put = (off, s) => h.set(enc.encode(s), off);
  if (name.length > 99) throw new Error(`tar name too long: ${name}`);
  put(0, name);
  put(100, "0000644\0");                       // mode
  put(108, "0000000\0");                       // uid
  put(116, "0000000\0");                       // gid
  put(124, size.toString(8).padStart(11, "0") + "\0");
  put(136, Math.floor(mtime / 1000).toString(8).padStart(11, "0") + "\0");
  put(148, "        ");                        // checksum field is spaces while the sum is taken
  put(156, "0");                               // a plain file
  put(257, "ustar\0" + "00");
  let sum = 0;
  for (const b of h) sum += b;
  put(148, sum.toString(8).padStart(6, "0") + "\0 ");
  return h;
}

/**
 * files: [{ name, bytes }] where bytes is a Uint8Array or a Blob. Returns the parts of the archive in order,
 * for `new Blob(parts)`, which takes either. Nothing is copied here and a Blob part is not even read: the
 * archive references the frame's blob, so building it does not put the run in memory a second time. That is
 * the whole point at 2160, where a run is 2.8 GB and the phone is already short by the time it is asked.
 */
export function tarParts(files, mtime = Date.now()) {
  const parts = [];
  for (const { name, bytes } of files) {
    const size = bytes.length ?? bytes.size;   // Uint8Array or Blob
    parts.push(header(name, size, mtime));
    parts.push(bytes);
    if (pad(size)) parts.push(new Uint8Array(pad(size)));
  }
  parts.push(new Uint8Array(2 * BLOCK));       // the end of archive is two zero blocks
  return parts;
}

export const tarBlob = (files, mtime) => new Blob(tarParts(files, mtime), { type: "application/x-tar" });
