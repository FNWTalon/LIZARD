// ai: The wasm binding end to end (2026-10-03), node --test from bindings/wasm:
// ai:   1. the format arithmetic against the web sender's own (sim/lizard_pick.mjs), every row of the C test's table,
// ai:      where liblizard/build.sh has built the codec's wasm (else skipped, saying so);
// ai:   2. the test stream in every ring, one code and two (a 2:1 canvas read by its halves), every block back, none bad;
// ai:   3. the C test's 5 MB file (its generator, byte for byte) through two codes into a memory store: whole, its bytes,
// ai:      and its root the native library's (30859af3...: tests/roundtrip.c under LIZ_TEST_MB=5);
// ai:   4. where LIZ_NATIVE_DUMP names a directory of the native test's --dump frames: this build's paint of the same
// ai:      frames, the border to the byte and the picture within one grey level, and both read to the same blocks.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import * as liz from "../index.mjs";

await liz.init();

test("version and vector paths", () => {
  assert.equal(liz.abi(), 1);
  assert.match(liz.version(), /^\d+\.\d+\.\d+$/);
  assert.equal(liz.simd(), true, "the codec's vector paths in the wasm build");
});

// ai: the web sender's arithmetic is liblizard/sim's, over the codec's own wasm (build/ob.mjs, liblizard/build.sh's), not
// ai: the wasm preset's
const sim = new URL("../../../sim/", import.meta.url);
const noSim = !existsSync(new URL("lizard_pick.mjs", sim)) ? "no liblizard/sim/ (a copy of the binding without the tree)"
  : !existsSync(new URL("../build/ob.mjs", sim)) ? "no liblizard/build/ob.mjs: run liblizard/build.sh" : false;

test("the format arithmetic is the web sender's", { skip: noSim }, async () => {
  await (await import(new URL("ob.mjs", sim))).init();
  const { ROOM_FOR, pickVersion, MODULES, OB_QUIET, N_FOR } = await import(new URL("lizard_pick.mjs", sim));
  let rows = 0;
  for (let r = 0; r < 4; r++)
    for (let b = 1; b <= 128; b++) {
      assert.equal(liz.roomFor(b, r), ROOM_FOR(8 * b, r), `room for ${b} blocks, ring ${r}`);
      assert.equal(liz.geometry({ blocks: b, ring: r }).n, N_FOR(8 * b));
      rows++;
    }
  for (const codes of [1, 2])
    for (let r = 0; r < 4; r++)
      for (let w = 200; w <= 4000; w += 173)
        for (let h = 200; h <= 2200; h += 211) {
          const gf = codes > 1 ? ((codes - 1) * 12) / (MODULES(256, r) + 2 * OB_QUIET) : 0;
          const room = Math.max(64, Math.min(w / (codes + gf), h));
          assert.equal(liz.pick(w, h, { codes, ring: r }), pickVersion(room, 1024, r).subch / 8, `pick ${w} x ${h}`);
          rows++;
        }
  assert.ok(rows > 2000);
});

// ai: the frame on white, as a camera frame holds it (the C test's): a quarter taller than the frame, square for one code
// ai: (layout 1), 2:1 for two (layout 2)
function canvas(frame, g, codes = 2) {
  const ch = g.height + Math.floor(g.height / 4), cw = codes * ch, c = new Uint8Array(cw * ch).fill(255);
  const x0 = (cw - g.width) >> 1, y0 = (ch - g.height) >> 1;
  for (let y = 0; y < g.height; y++) c.set(frame.subarray(y * g.width, (y + 1) * g.width), (y0 + y) * cw + x0);
  return { c, cw, ch };
}

test("the test stream in every ring, one code and two", () => {
  for (const codes of [1, 2])
    for (let ring = 0; ring < 4; ring++) {
      // ai: a fixed first id: the C's own misses of a clean frame (about 1 in 200) are measured elsewhere
      const s = new liz.Sender({ test: true, firstId: 1 + 1000 * ring + 100 * codes }, { blocks: 26, ring, codes });
      const r = new liz.Receiver({ layout: codes });
      for (let i = 0; i < 3; i++) {
        const { data } = s.frame("grey");
        const g = s.geometry, img = canvas(data, g, codes);
        const got = r.push(img.c, img.cw, img.ch, { fmt: "grey" });
        const blocks = got.decoded.reduce((n, d) => n + d.count, 0);
        assert.equal(blocks, g.frameBlocks, `ring ${ring}, ${codes} codes, frame ${i}: every block`);
        assert.ok(got.verdict.test && got.verdict.bad === 0, "judged the test stream's");
        assert.equal(r.decoder.held, 26, "the held word");
      }
      s.free(); r.free();
    }
});

// ai: tests/roundtrip.c's generator: xorshift64 (13, 7, 17), a byte the state's bits 32 to 39
function cBytes(n) {
  const out = new Uint8Array(n), M = (1n << 64n) - 1n;
  let s = 0x9e3779b97f4a7c15n;
  for (let i = 0; i < n; i++) {
    s ^= (s << 13n) & M; s ^= s >> 7n; s ^= (s << 17n) & M;
    out[i] = Number((s >> 32n) & 0xffn);
  }
  return out;
}

test("a 5 MB file through two codes, the native library's root", () => {
  const bytes = cBytes(5000000);
  const s = new liz.Sender({ bytes, name: "roundtrip.bin", type: "application/octet-stream" }, { blocks: 60, codes: 2 });
  const r = new liz.Receiver({ layout: 2 });
  assert.equal(s.info.root, "30859af3062484f3ecb237460cf24e8a44a88a8cd4b1431d02d9d0d9934a202f", "the sender's root, the native's");
  let frames = 0;
  for (; frames < 400 && !r.file; frames++) {
    const { data } = s.frame("grey");
    const { c, cw, ch } = canvas(data, s.geometry);
    const got = r.push(c, cw, ch, { fmt: "grey" });
    assert.ok(!got.verdict.test && !got.verdict.bad);
  }
  const f = r.file;
  assert.ok(f, `whole after ${frames} frames`);
  assert.equal(f.root, s.info.root);
  assert.equal(f.name, "roundtrip.bin");
  assert.deepEqual(Buffer.from(f.bytes), Buffer.from(bytes));
  s.free(); r.free();
});

test("this build's paint against the native build's", { skip: !process.env.LIZ_NATIVE_DUMP }, () => {
  const dir = process.env.LIZ_NATIVE_DUMP;
  for (const codes of [1, 2])
    for (let ring = 0; ring < 4; ring++) {
      const native = new Uint8Array(readFileSync(`${dir}/paint-26-${ring}-${codes}.grey`));
      const s = new liz.Sender({ test: true, firstId: 1 }, { blocks: 26, ring, codes }), g = s.geometry;
      const mine = s.frame("grey").data;
      assert.equal(mine.length, native.length);
      // ai: the picture square of each code: its first sample (QUIET + 15 modules) to span modules on
      const inPicture = (x, y) => {
        const k = Math.floor(x / (g.side + g.gap)), sx = x - k * (g.side + g.gap), a = (2 + 15) * g.pxm, b = a + g.span * g.pxm;
        return sx < g.side && sx >= a && sx < b && y >= a && y < b;
      };
      let border = 0, picture = 0, worst = 0;
      for (let y = 0; y < g.height; y++)
        for (let x = 0; x < g.width; x++) {
          const d = Math.abs(mine[y * g.width + x] - native[y * g.width + x]);
          if (!d) continue;
          if (inPicture(x, y)) { picture++; worst = Math.max(worst, d); } else border++;
        }
      assert.equal(border, 0, `ring ${ring}, ${codes} codes: the border to the byte`);
      assert.ok(worst <= 1, `ring ${ring}, ${codes} codes: the picture within a level (${picture} pixels off, at most ${worst})`);
      const d = new liz.Decoder(), held = { held: 0 };
      const read = (img) => {
        const { c, cw, ch } = canvas(img, g, codes);
        return liz.layoutRects(cw, ch, codes).map((q) => d.decode(c.subarray(q.y * cw + q.x), q.w, q.h, { fmt: "grey", stride: cw, held }));
      };
      const a = read(mine), b = read(native);
      assert.deepEqual(a.map((x) => Buffer.from(x.blocks)), b.map((x) => Buffer.from(x.blocks)), "the same blocks from both");
      assert.equal(a.reduce((n, x) => n + x.count, 0), g.frameBlocks);
      s.free(); d.free();
    }
});
