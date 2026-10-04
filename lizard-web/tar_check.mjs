// tar.mjs writes archives that real tar reads. node lizard-web/tar_check.mjs
//
// The page builds a recorded run into one of these and hands it to the browser as a download, so if this is
// wrong a run is unrecoverable and there is nothing on the rig to fall back on: the frames only ever existed
// in the phone's memory.
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { tarParts } from "./tar.mjs";

let fail = 0;
const bad = (m) => { console.log(`  FAIL ${m}`); fail++; };
const dir = mkdtempSync(join(tmpdir(), "tarcheck-"));

// A run's shape: a meta, a log, and frames whose sizes are deliberately not multiples of 512.
const files = [
  { name: "meta.json", bytes: new TextEncoder().encode(JSON.stringify({ w: 1080, h: 1080 })) },
  { name: "stats.jsonl", bytes: new TextEncoder().encode('{"a":1}\n{"a":2}\n') },
  { name: "0000.gray", bytes: Uint8Array.from({ length: 1000 }, (_, i) => i & 255) },
  { name: "0001.gray", bytes: Uint8Array.from({ length: 1024 }, (_, i) => (i * 7) & 255) },
  { name: "0002.gray", bytes: new Uint8Array(0) },
];
const parts = tarParts(files, 1700000000000);
const total = parts.reduce((t, p) => t + p.length, 0), flat = new Uint8Array(total);
let o = 0; for (const p of parts) { flat.set(p, o); o += p.length; }
writeFileSync(join(dir, "run.tar"), flat);

if (total % 512) bad(`archive is ${total} bytes, not a multiple of 512`);

try {
  const listed = execFileSync("tar", ["-tf", join(dir, "run.tar")], { encoding: "utf8" }).trim().split("\n");
  console.log(`tar -tf lists ${listed.length} entries: ${listed.join(" ")}`);
  if (listed.length !== files.length) bad(`listed ${listed.length} entries, wrote ${files.length}`);
  for (const f of files) if (!listed.includes(f.name)) bad(`${f.name} missing from the listing`);

  execFileSync("tar", ["-xf", join(dir, "run.tar"), "-C", dir]);
  for (const f of files) {
    const got = new Uint8Array(readFileSync(join(dir, f.name)));
    if (got.length !== f.bytes.length) { bad(`${f.name}: extracted ${got.length} bytes, wrote ${f.bytes.length}`); continue; }
    for (let i = 0; i < got.length; i++) if (got[i] !== f.bytes[i]) { bad(`${f.name}: byte ${i} differs`); break; }
  }
  console.log(`tar -xf returned every file byte for byte`);
} catch (e) {
  bad(`tar refused the archive: ${e.message}`);
}

// The phone does not hand in arrays: a recorded frame arrives from its worker as a Blob and goes into the
// archive as one, so the header's size field is read from .size there and not .length. Same files, same bytes.
const asBlobs = files.map((f) => ({ name: f.name, bytes: new Blob([f.bytes]) }));
const blobTar = new Blob(tarParts(asBlobs, 1700000000000));
if (blobTar.size !== total) bad(`blob parts made a ${blobTar.size} byte archive, arrays made ${total}`);
else {
  const got = new Uint8Array(await blobTar.arrayBuffer());
  const at = got.findIndex((b, i) => b !== flat[i]);
  if (at >= 0) bad(`blob parts differ from array parts at byte ${at}`);
  else console.log("a run built from Blobs is byte for byte the archive built from arrays");
}

// A name that cannot fit has to be refused rather than silently truncated into a different file.
try { tarParts([{ name: "x".repeat(120), bytes: new Uint8Array(1) }]); bad("accepted a name too long for the header"); }
catch { console.log("a name too long for the header is refused"); }

rmSync(dir, { recursive: true, force: true });
console.log(fail ? `\n${fail} FAILED` : "\nall checks passed");
process.exit(fail ? 1 : 0);
