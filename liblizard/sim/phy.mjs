// One shape for every code under test, so the harness cannot treat them differently.
//
//   const phy = await makePhy(spec)
//   phy.frame(seq, baseId)         -> { w, h, quiet, dark | drive, truth }   what the screen shows
//     baseId is the first block id in the frame, defaulting to seq * blocksPerFrame. A live
//     sender passes its own monotonic counter instead: blocksPerFrame moves when the layout is
//     re-picked mid-stream, and an id derived from it would walk backwards onto ids already sent.
//   phy.decode(img, iw, ih, truth, rgba) -> one result per phy.variants entry (rgba: img is RGBA, not luma):
//        { fresh, seen, bad, found, ber, mi, ms: { detect, sample, fec, total } }
//
// A variant is a decoder setting applied to the same capture, which is how ablations stay
// paired. `fresh` counts fountain blocks not seen before, `seen` every valid block in the
// capture. usefulBytes is what one block hands the fountain: headers and ids are not payload.
import { init as initOb, Focus, FocusAny, prof, streamBlock } from "./ob.mjs";
import { isControlId, PAYLOAD } from "./xfer.mjs";
import { PICTURE_SIZES } from "./lizard_pick.mjs";

// What the sender paints outside a LIZARD frame, in SAMPLES: the codec's margin (fc.quiet modules, src/focus.h
// FOCUS_QUIET), or spec.quiet modules where an experiment overrides it. Specs give modules; the painters take samples,
// and until 2026-09-23 this file handed the margin over unconverted, so the rig painted half a module where the
// harnesses painted two.
const marginSamples = (spec, fc) => (spec.quiet ?? fc.quiet) * fc.cell;

// ai: LDPC rate indices to names (src/ldpc.h): the format's four
const RATE = { 2: "1/2", 3: "2/3", 4: "3/4", 6: "7/8" };

// Block contents are a function of the block id, so a receiver can check any block it decodes
// without being told which frame it came from.
//
// The xorshift32 walk below is the stream a spec that names none gets: the simulator's harnesses paint
// and check their own frames with it. Every recording kept is SHAKE256's (below), which a live sender
// paints. Streams say which generator they used; no field means this one.
export function blockBytesLegacy(id, len, out = new Uint8Array(len)) {
  let s = (Math.imul(id + 1, 0x9e3779b1) ^ 0x85ebca6b) >>> 0;
  for (let i = 0; i < len; i++) { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; out[i] = s & 255; }
  return out;
}

// SHAKE256 of the id (sim/ob.mjs, src/shake.c). xorshift32 has a single orbit of 2^32 - 1, so
// two ids whose seeds land within a block of each other on it emit byte strings that are shifts
// of one another: 119 such pairs in 40k ids, which is the birthday rate for a 316-byte block and
// so nothing the seeding could fix. One id seeds the absorbing zero state and emits an all-zero
// block. Neither happens under a hash, which is what makes the measured tables defensible.
export function blockBytesShake(id, len, out = new Uint8Array(len)) {
  out.set(streamBlock(id, len));
  return out;
}

// The spec a RECORDING was made with, fit to replay: every recording kept names its span, mode 1 and the bit map
// (src/focus.h FOCUS_BITMAP), so it replays as written (the rules for the specs of 2026-09-23, before the map and the
// spans were written down, went with those recordings).
export const recordedSpec = (spec) => spec;

// ai: The largest picture a receiver builds: the ladder's top (1536, LIZARD-576 to -1024), so it reads any version the
// ai: word names, with no flag (2026-09-24; 1024 until 2026-09-29, the receiver's #sizes menu choosing 1536, the menu
// ai: deleted that day since the ring's word names the size). A blind C worker builds a picture's
// ai: codec only when a word names it (src/wasm.c pic_for, a few ms since the same day's counting sort).
export const BLIND_NMAX = PICTURE_SIZES.at(-1);
// ai: The blind spec, the one a receiver decodes LIZARD with: any ring, any picture to nmax, told nothing.
export const blindSpec = (nmax = BLIND_NMAX) => ({ phy: "focus", blind: 1, nmax });
// ai: A recording read blind, as a receiver would have read it; stream is the recording's generator, for blockJudge.
// ai: Only a recording on today's border reads: research/captures v0.1 and v0.2 were painted before the three rings
// ai: (2026-09-27), and scripts/exp/recordings.mjs lists the current version's runs alone.
export const recordedBlind = (spec) => ({ ...blindSpec(Math.max(BLIND_NMAX, spec.n ?? 0)), stream: spec.stream });

// ai: name -> generator. A spec names its stream so a harness and any later replay agree (a recording carries its
// ai: sender's spec); the live test stream is SHAKE256 (SPEC 9.3), which a blind receiver judges from the light alone
// ai: (blockJudge).
export const STREAMS = { legacy: blockBytesLegacy, shake256: blockBytesShake };
export const sourceFor = (stream) => {
  const gen = STREAMS[stream ?? "legacy"];
  if (!gen) throw new Error(`unknown test stream ${stream}`);
  return (id, out) => gen(id, out.length, out);
};

// ai: A blind receiver's judge of a frame's verified blocks, laid out as the codec lays them (the 4-byte id, then the
// ai: payload), from the light alone (the whole receiver blind since 2026-09-26). The test stream's payload
// ai: is a function of its id (SHAKE256 of it, SPEC 9.3), so a frame is the test stream's when one of its verified data
// ai: blocks carries its id's bytes (a file's block does so with probability 2^-3752); every other verified data block
// ai: of that frame is then bad (the CRC let a wrong block through), counted and dropped. A frame with none is a file's,
// ai: or a test frame whose every verified block is wrong, the one case not counted: the CRC alone judges it.
// ai: frame(ok, blocks): ok[k] set for each verified block k, at k * (payload + 4). Returns got, the frame's every block
// ai: not judged bad, control blocks (a transfer's header or manifest) included, with their bytes; seen (got's count),
// ai: bad; test, whether the frame was judged the test stream's; judged, its data blocks held to the stream.
// ai: The judge keeps nothing from frame to frame (2026-09-29): which blocks are new is the receiver's page's to say, the
// ai: one set of ids that a new transfer or Clear lets go (lizard-web/recv.mjs). Until then each decode worker's judge kept the
// ai: ids it had reported and sent only new ones, a set nothing let go, so after Clear, or at a second file in one
// ai: session (whose ids start again from 0), the blocks the workers had already reported never reached the page again.
// ai: stream: the generator (sim/phy.mjs STREAMS), a replay's recording naming its own.
export function blockJudge(payload = PAYLOAD, stream = "shake256") {
  const B = payload + 4, want = new Uint8Array(payload), gen = sourceFor(stream);
  return {
    frame(ok, blocks) {
      let test = false;
      const got = [], data = [];
      let bad = 0;
      for (let k = 0; k < ok.length; k++) {
        if (!ok[k]) continue;
        const o = k * B, id = (blocks[o] | (blocks[o + 1] << 8) | (blocks[o + 2] << 16) | (blocks[o + 3] << 24)) >>> 0, bytes = blocks.slice(o + 4, o + B);
        if (isControlId(id)) { got.push({ id, bytes }); continue; }
        gen(id, want);
        let same = true;
        for (let q = 0; q < payload; q++) if (bytes[q] !== want[q]) { same = false; break; }
        if (same) test = true;
        data.push({ id, bytes, same });
      }
      for (const d of data) if (test && !d.same) bad++; else got.push({ id: d.id, bytes: d.bytes });
      return { got, seen: got.length, bad, test, judged: test ? data.length : 0 };
    },
  };
}

// ai: The blind receiver in the C (sim/ob.mjs FocusAny): it registers against every ring, reads the word there and
// ai: decodes the picture the word names, to spec.nmax. It holds the last word it read and decodes a wordless frame
// ai: with it (since 2026-09-27: the ring bootstraps the word, and a frame whose word does not decode leaves the held
// ai: one as it is): the one piece of state that carries from frame to frame, so frames go through one of these in the
// ai: order they were captured. Nothing else carries. Blocks are judged from the light
// ai: (blockJudge); spec.stream names a replay's generator, SHAKE256 when left out.
async function makeBlind(spec) {
  await initOb();
  const nmax = spec.nmax ?? BLIND_NMAX, fa = new FocusAny(nmax);
  const judge = blockJudge(fa.blockBytes - 4, spec.stream ?? "shake256");
  let held = 0, builds = 0;
  return {
    label: `focus blind to ${nmax}`, family: "focus", variants: ["blind"], usefulBytes: fa.blockBytes - 4, nmax,
    decode(img, iw, ih, truth, rgba = false) {
      const t = performance.now(), r = fa.decode(img, iw, ih, { mesh: 2, held, rgba }), total = performance.now() - t;
      if (r.fmt) held = r.fmt.version;
      // ai: built: this frame built a picture's codec (its first sight of that ring and picture; src/wasm.c pic_for).
      const built = r.builds > builds;
      builds = r.builds;
      return [{ ...judge.frame(r.ok, r.blocks), found: r.found, n: r.pictureN, ring: r.ring, held: r.held, fmt: r.fmt, quad: r.quad, built, ber: NaN, mi: NaN, ms: { detect: r.msDetect, total } }];
    },
    reset() { held = 0; },
    prof, free() { fa.free(); },
  };
}
// variants: [{ name, gamma, mesh }]. Each block starts with its 4-byte fountain id.
async function makeFocus(spec) {
  await initOb();
  const { n = 512, subch = 32, mode = 1, clip = 2, span = 0, tilt = 0, tiers = null, bitmap, fps = 0, variants = [{ name: "ldpc" }] } = spec;
  const fc = new Focus(n, subch, mode, { clip, span, tilt, tiers, bitmap }), B = fc.blockBytes, T = fc.blocks, seen = variants.map(() => new Set()), buf = new Uint8Array(T * B), want = new Uint8Array(B - 4);
  const quiet = marginSamples(spec, fc);
  // The page's pixels come from the codec's own paint, which has only its margin: an experiment's override has no way there.
  const noRGBA = spec.quiet !== undefined && spec.quiet !== fc.quiet ? `spec.quiet ${spec.quiet} is the simulator's override; the codec paints ${fc.quiet}` : null;
  if (fps) fc.setFps(fps);   // the display rate goes in the band (src/fmt.h); 0 states none, which is the default
  let source = sourceFor(spec.stream);
  return {
    label: `focus ${n} ${tiers ? tiers.map(([r, b]) => `${b}x${RATE[r]}`).join("+") : `${subch}sub`}${span ? " span" + span : ""}${tilt ? " tilt " + tilt + " dB" : ""}`, family: "focus", variants: variants.map((v) => v.name), usefulBytes: B - 4, blocksPerFrame: T, modules: fc.side, bitmap: fc.bitmap,
    // ai: the paint's samples a module and the codec's margin in modules (src/focus.h FOCUS_QUIET), for a painter
    // ai: that places symbols in modules (lizard-web/send-worker.mjs: two codes a gap apart, 2026-09-30)
    cell: fc.cell, quiet: fc.quiet,
    // baseId may instead be the frame's T ids, for a sender whose ids are not a run (sim/xfer.mjs: chunk and symbol).
    frame(seq, baseId = seq * T) {
      for (let t = 0; t < T; t++) { const id = typeof baseId === "number" ? baseId + t : baseId[t], o = t * B; buf[o] = id & 255; buf[o + 1] = (id >>> 8) & 255; buf[o + 2] = (id >>> 16) & 255; buf[o + 3] = id >>> 24; source(id, buf.subarray(o + 4, o + B)); }
      fc.setParity(0);
      return { w: fc.side, h: fc.side, quiet, drive: fc.encode(buf), truth: null };
    },
    // The same frame as the pixels a page paints, margin included (lizard-web/send-worker.mjs): { w, rgba }, the rgba a view
    // into the wasm heap that the next call overwrites. ai: seq is the painted count, so seq mod 4 is the pilots'
    // (SPEC 7.3): a page's frames flash (frame() above, the simulator's, paints count 0, as before the pilots).
    frameRGBA(seq, baseId = seq * T) {
      if (noRGBA) throw new Error(noRGBA);
      fc.setParity(seq & 3);
      return fc.encodeRGBA(this.frameBlocks(seq, baseId));
    },
    // ai: The frame's T blocks alone, each its id (little-endian) then its payload, B bytes a block, for an encoder
    // ai: elsewhere (the sender's GPU encoder, gpu/encoder.mjs). A view reused by the next call.
    frameBlocks(seq, baseId = seq * T) {
      for (let t = 0; t < T; t++) { const id = typeof baseId === "number" ? baseId + t : baseId[t], o = t * B; buf[o] = id & 255; buf[o + 1] = (id >>> 8) & 255; buf[o + 2] = (id >>> 16) & 255; buf[o + 3] = id >>> 24; source(id, buf.subarray(o + 4, o + B)); }
      return buf;
    },
    // One decoder result turned into the harness's, shared by the one-call path and the split one so the two
    // cannot report differently.
    tally(r, vi, total) {
      const got = [];
      let fresh = 0, n2 = 0, bad = 0;
      for (let k = 0; k < T; k++) {
        if (!r.ok[k]) continue;
        const o = k * B, id = (r.blocks[o] | (r.blocks[o + 1] << 8) | (r.blocks[o + 2] << 16) | (r.blocks[o + 3] << 24)) >>> 0;
        // A transfer's header or manifest block (sim/xfer.mjs) is no test stream's, so it is not compared: its CRC is
        // the judge, as for a file. Not fresh, since it is not payload, and handed on every time it is read, since its
        // id is the same in every transfer and only its bytes say whether it is a new one.
        if (isControlId(id)) { n2++; got.push({ id, bytes: r.blocks.slice(o + 4, o + B) }); continue; }
        // With no source to compare against (the receiving end of a file), the block's own CRC,
        // checked in the decoder, is the only judge.
        let same = true;
        if (source) { source(id, want); for (let q = 0; q < B - 4; q++) if (r.blocks[o + 4 + q] !== want[q]) { same = false; break; } }
        if (!same) { bad++; continue; }
        n2++;
        if (!seen[vi].has(id)) { seen[vi].add(id); fresh++; got.push({ id, bytes: r.blocks.slice(o + 4, o + B) }); }
      }
      return { got, fresh, seen: n2, bad, found: r.found, fmt: r.fmt, quad: r.quad, mark: r.mark, ber: NaN, mi: NaN, ms: { detect: r.msDetect, sample: r.msSample, fec: r.msDecode, total } };
    },
    decode(img, iw, ih, truth, rgba = false) {
      return variants.map((v, vi) => {
        const t = performance.now(), r = fc.decode(img, iw, ih, { ...v, rgba }), total = performance.now() - t;
        return this.tally(r, vi, total);
      });
    },
    reset() { seen.forEach((s) => s.clear()); },
    seenCount: () => seen.reduce((t, x) => t + x.size, 0),
    setSource(fn) { source = fn; },
    // The rate the band states (src/fmt.h). A sender may call it whenever its own target moves.
    setFps(v) { fc.setFps(v); },
    prof, free() { fc.free(); },
  };
}

export async function makePhy(spec) {
  if (spec.phy === "focus") return spec.blind ? makeBlind(spec) : makeFocus(spec);
  throw new Error(`unknown phy ${spec.phy}`);
}
