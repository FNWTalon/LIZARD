// ai: The self-test's view of a batch (scripts/pages/gpu_selftest.mjs): what every stage left in its buffers, copied out in
// ai: the batch's own encoder after its passes and read back beside the batch's readback. Only a decoder made with
// ai: probe: true has one (FrontHalf.create), and only a batch run with probe: true uses it; the receiver asks for
// ai: neither. Such a decoder gives the pyramid levels, the accumulator and the cascade's logits and list COPY_SRC, a
// ai: buffer flag no stage reads. The input layer keeps the receiver's usage flags (a texture's can change its layout):
// ai: it is read through textureLoad, as every stage reads it, into a buffer of packed bytes.
import { DIMS } from "./wgsl/common.mjs";
import { SCORE, HYP } from "./wgsl/finder.mjs";

const U = globalThis.GPUBufferUsage ?? {};

// ai: SCORE (F6a) again after the batch, on its final buffers, into a scratch score: if the batch's scores and these
// ai: disagree, the batch's SCORE read other inputs than GATHER left (a barrier missing between them). Each hypothesis
// ai: also writes 8 vec4f of what it computed on the way to its score:
// ai:   [0] live, frame valid, quad count, H[8] (thread 0)   [1] H[0..3]   [2] H[4..7]   [3] live, H[0], H[4], H[8] (thread 63)
// ai:   [4] module size, level, line, gap   [5] pair sum, points read, contrast, 0
// ai:   [6] the TL mark centre through H (x, y), look() there on levels 0 and 1   [7] track sum, track points, the ungated score, 0
// ai: Built from SCORE's own text at four anchors, so it is the shader the batch runs; a missing anchor throws.
const R = `(f * ${HYP}u + hyp) * 8u`;
const SCORE_PROBE = [
  ["var<workgroup> ok: u32;\n", "var<workgroup> ok: u32;\n@group(0) @binding(11) var<storage, read_write> dbg: array<vec4f>;\n"],
  ["  let live = workgroupUniformLoad(&ok);\n", `  let live = workgroupUniformLoad(&ok);
  if (li == 0u) { dbg[${R}] = vec4f(f32(live), f32(frames[f].valid), qm.w, H[8]); dbg[${R} + 1u] = vec4f(H[0], H[1], H[2], H[3]); dbg[${R} + 2u] = vec4f(H[4], H[5], H[6], H[7]); }
  if (li == 63u) { dbg[${R} + 3u] = vec4f(f32(live), H[0], H[4], H[8]); }
`],
  ["  let con = lev.y - lev.x;\n", `  let con = lev.y - lev.x;
  if (li == 0u) {
    dbg[${R} + 4u] = vec4f(u, f32(l), lev.x, lev.y);
    dbg[${R} + 5u] = vec4f(lev.z, sums[0].w, con, 0.0);
    let pc = applyH(Hl, vec2f(8.0, 8.0));
    dbg[${R} + 6u] = vec4f(pc, look(0u, f, pc), look(1u, f, pc));
  }
`],
  [`  if (li == 0u) { score[f * ${HYP}u + hyp] = select(`, `  if (li == 0u) { dbg[${R} + 7u] = vec4f(sums[0].x, sums[0].y, sums[0].x / max(sums[0].y, 1.0), 0.0); }
  if (li == 0u) { score[f * ${HYP}u + hyp] = select(`],
].reduce((src, [at, to]) => {
  if (!src.includes(at)) throw new Error(`probe: SCORE no longer has ${JSON.stringify(at)}`);
  return src.replace(at, () => to);
}, SCORE);

// ai: One invocation a word: four pixels of frame g.y, row-major at the frame's own width, as round(255 v).
const READ_INPUT = DIMS + /* wgsl */ `
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> frames: array<Frame>;
@group(0) @binding(2) var<storage, read_write> words: array<u32>;
@group(0) @binding(3) var<uniform> P: vec4u;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let f = g.y;
  if (k >= P.x) { return; }
  let F = frames[f];
  var word = 0u;
  if (F.valid != 0u) {
    let n = F.w * F.h;
    for (var b = 0u; b < 4u; b++) {
      let i = 4u * k + b;
      if (i < n) {
        let v = textureLoad(img, vec2i(i32(i % F.w), i32(i / F.w)), i32(f), 0).r;
        word = word | ((u32(round(clamp(v, 0.0, 1.0) * 255.0)) & 255u) << (8u * b));
      }
    }
  }
  words[f * P.x + k] = word;
}
`;

export class Probe {
  static async create(fh) {
    const p = new Probe();
    p.input = await fh.pipeline(READ_INPUT, ["tex", "ro", "rw", "uniform"]);
    p.score = await fh.pipeline(SCORE_PROBE, ["tex", "ro", "ro", "ro", "ro", "ro", "uniform", "ro", "ro", "rw", "uniform", "rw"]);
    return p;
  }

  // ai: Called on the lane that runs the batch, with its encoder, once its passes have ended; nb frames. Returns the
  // ai: batch's reader: start() right after the submit, read() once the batch's readback is in.
  encode(ln, enc, nb) {
    const d = ln.device, B = ln.B, words = Math.ceil((ln.W * ln.H) / 4);
    const inBuf = d.createBuffer({ size: 4 * words * nb, usage: U.STORAGE | U.COPY_SRC });
    const uni = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(uni, 0, new Uint32Array([words, 0, 0, 0]));
    const group = d.createBindGroup({ layout: this.input.bgl, entries: [{ binding: 0, resource: ln.texView }, { binding: 1, resource: { buffer: ln.framesBuf } },
      { binding: 2, resource: { buffer: inBuf } }, { binding: 3, resource: { buffer: uni } }] });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.input.p);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(words / 256), nb);
    let score2 = null, dbg = null;
    if (ln.quadsBuf) {
      score2 = d.createBuffer({ size: 4 * HYP * nb, usage: U.STORAGE | U.COPY_SRC });
      dbg = d.createBuffer({ size: 16 * 8 * HYP * nb, usage: U.STORAGE | U.COPY_SRC });
      const entries = [{ binding: 0, resource: ln.texView }, ...ln.level.map((lv, i) => ({ binding: 1 + i, resource: { buffer: lv.buf } })),
        { binding: 5, resource: { buffer: ln.framesBuf } }, { binding: 6, resource: { buffer: ln.ldUni } },
        ...[ln.quadsBuf, ln.planBuf, score2, ln.planUni, dbg].map((buffer, k) => ({ binding: 7 + k, resource: { buffer } }))];
      pass.setPipeline(this.score.p);
      pass.setBindGroup(0, d.createBindGroup({ layout: this.score.bgl, entries }));
      pass.dispatchWorkgroups(HYP * nb);
    }
    pass.end();
    // ai: [name, buffer, bytes a frame, element type]: every buffer holds B equal frame strides.
    const per = (b) => b.size / B;
    const parts = [["input", inBuf, 4 * words, Uint8Array], ...ln.level.map((lv, i) => [`level${i + 1}`, lv.buf, per(lv.buf), Float32Array]),
      ["counts", ln.countsBuf, per(ln.countsBuf), Uint32Array], ["peaks", ln.peaksBuf, per(ln.peaksBuf), Float32Array],
      ["readings", ln.readingsBuf, per(ln.readingsBuf), Float32Array], ["acc", ln.accBuf, per(ln.accBuf), Uint32Array],
      ["cand", ln.candBuf, per(ln.candBuf), Float32Array], ["quads", ln.quadsBuf, per(ln.quadsBuf), Float32Array],
      ["score", ln.scoreBuf, per(ln.scoreBuf), Float32Array], ["result", ln.resultBuf, per(ln.resultBuf), Float32Array],
      ["maps", ln.mapsBuf, per(ln.mapsBuf), Float32Array],
      // ai: F7's working: meas (a node: image x, y, correlation, 1 where measured), resid after the side fits (a node:
      // ai: the fitted residual x, y, then the measured residual x, y) and REFIT's debug words (wgsl/register.mjs
      // ai: REFIT_DBG, 63 floats a frame: the first iteration's theta 0-8, normal matrix 9-53 and gradient 54-62), so a
      // ai: frame's registration can be held against another's.
      ["meas", ln.measBuf, per(ln.measBuf), Float32Array], ["resid", ln.residBuf, per(ln.residBuf), Float32Array], ["refit", ln.refitDbg, per(ln.refitDbg), Float32Array],
      ...(ln.logitBuf ? [["logits", ln.logitBuf, per(ln.logitBuf), Float32Array]] : []), ...(ln.listBuf ? [["list", ln.listBuf, per(ln.listBuf), Uint32Array]] : []),
      ...(score2 ? [["score2", score2, 4 * HYP, Float32Array], ["scoreDbg", dbg, 16 * 8 * HYP, Float32Array]] : [])];
    let off = 0;
    const at = parts.map(([, , bytes]) => { const o = off; off += bytes * nb; return o; });
    const readBuf = d.createBuffer({ size: off, usage: U.MAP_READ | U.COPY_DST });
    parts.forEach(([, buf, bytes], k) => enc.copyBufferToBuffer(buf, 0, readBuf, at[k], bytes * nb));
    const shape = { W: ln.W, H: ln.H, gx: ln.gx, gy: ln.gy, cap: ln.cap, keep: ln.cascade?.keep ?? 0, levels: ln.level.map((lv) => ({ stride: lv.w, rows: lv.h })) };
    let mapped = null;
    const drop = () => { inBuf.destroy(); uni.destroy(); readBuf.destroy(); score2?.destroy(); dbg?.destroy(); };
    return {
      start() { mapped = readBuf.mapAsync(GPUMapMode.READ); mapped.catch(() => {}); },
      discard() { (mapped ?? Promise.resolve()).then(drop, drop); },
      // ai: Per frame: { shape, <part>: a typed array of that frame's stride } (input is its bytes, 4 a word).
      async read() {
        try {
          await mapped;
          const all = readBuf.getMappedRange();
          const frames = Array.from({ length: nb }, () => ({ shape }));
          parts.forEach(([name, , bytes, T], k) => {
            for (let f = 0; f < nb; f++) frames[f][name] = new T(all.slice(at[k] + f * bytes, at[k] + (f + 1) * bytes));
          });
          readBuf.unmap();
          return frames;
        } finally { drop(); }
      },
    };
  }
}
