// ai: cnn-v3, the cascade's small classifier (role front.twins.small), int8, on the device's cooperative-matrix units
// ai: (core/nets/README.md): a GLSL kernel (core/nets/cls_v3.comp over core/nets/cls.glsl) that replaces the web's
// ai: WGSL twin (liblizard/gpu/wgsl/classify_gemm.mjs, int8 form) over the same bind group layout and dispatch, and
// ai: computes the same contract (liblizard/gpu/cnn/q8c.mjs forwardCodesC on the kernel's own Xq) with every conv and
// ai: fc1 a GEMM on coopMatMulAdd (int8 x int8 -> int32, K steps of 32). This module packs the q8 file for the kernel
// ai: (the weights buffer's bytes, below), checks the packing by running the kernel's GEMM formulation on the CPU
// ai: against forwardCodesC (random patches, every code, fc1 and the outputs), compiles the GLSL with glslang and
// ai: spirv-val (glslang's --spirv-val), and returns the kernel's manifest entry (gen/nets.mjs).
// ai:
// ai: Not in the manifest by default (LIZ_NETS_CLS=1 builds the kernel and returns its entry): the fast build's premise
// ai: (the device's element layout and stride unit) is checked by the kernel, which writes no-mark readings where it
// ai: fails, and the host adopts a listed kernel only where its first batch's outputs equal the WGSL twin's to the bit
// ai: (core/dec/front.cpp measureTwins), but it adds 1.5 to 1.8 s to a cold start and leans on a compiler with known
// ai: faults (core/nets/README.md). The large net (cnn-v1) has no kernel: its f16 form costs a tenth of a millisecond a
// ai: frame.
// ai:
// ai: The weights buffer (u32 words; every matrix 16-byte aligned, as coopMatLoad wants):
// ai:   per conv L: B_L, S_L K-steps of a 32 x N_L int8 matrix, row-major (row k, column n: byte (s 32 + k) N_L + n),
// ai:               then N_L biases (i32) and N_L requant scales (f32), one a GEMM column (a column is a channel, or
// ai:               (position j, channel) where a GEMM row holds several outputs: conv1 4 x 8);
// ai:   fc1:        A, 64 x K / 2 int8 row-major: row h 32 + o is output o over the half h of K (k = h K / 2 + k'),
// ai:               k in the last conv's NHWC order, (y 4 + x) C + c (torch's (c, y, x) permuted); then 32 biases (i32)
// ai:               and 32 dequant scales (f32). The kernel's fc1 is C = A B with B column 2 n + h' patch n's half h'
// ai:               (a column stride of K / 2 bytes over the patches' maps, K bytes apart): fc1 of patch n, output o
// ai:               is C[o][2 n] + C[32 + o][2 n + 1], the two diagonal blocks, so no row of A is padding;
// ai:   fc2:        the 5 x 32 weights (f32, row-major), then 5 biases (f32), padded to 8.
// ai: Each conv's GEMM (the A rows the kernel builds, core/nets/cls.glsl):
// ai:   c1     conv1, 1 -> 8 at stride 2: a row is 4 outputs (y, 4 xg .. 4 xg + 3), K 32 = the three input rows'
// ai:          8 columns 8 xg .. 8 xg + 7 (bytes 8 ky + c) and their left neighbour 8 xg - 1 (byte 24 + ky), N 32 =
// ai:          (j, co) at 8 j + co; 64 rows a patch (r = 4 y + xg)
// ai:   s2c8   stride 2 from 8 channels (conv2): a row an output, step ky = input row 2 y - 1 + ky, bytes
// ai:          8 kx + c for input column 2 x - 1 + kx (24 bytes, 8 zero), N = outC
// ai:   s2t2   stride 2 from 16 channels (conv3): a row an output, step s = taps 2 s and 2 s + 1 (t = 3 ky +
// ai:          kx; 9 taps in 5 steps), bytes 16 h + c for tap 2 s + h, N = outC
// ai: Every activation map is NHWC codes (a position's channels in consecutive bytes), so a conv's C row is its
// ai: output map's bytes in order.
//   node liblizard/gen/nets_cls.mjs check      the CPU check alone, no glslang
//   node liblizard/gen/nets_cls.mjs compare <dump A> <dump B> [frames]   two `lizard_gpu_check dump` runs held per peak
// ai: Env: LIZ_NETS_CLS=1 (above); LIZ_NETS_SHAPES, the shapes the kernel is built for (a device's
// ai: VkCooperativeMatrixPropertiesKHR, subgroup scope): adreno (the default: M 64, N 16 and 32, K 32, strides in
// ai: bytes; the S26 Ultra's Adreno 840) or nv16 (M 16, N 16, K 32, strides in words; the desktop's RTX 4090, for
// ai: checking the kernel's staging path there).
import { readFileSync, writeFileSync, mkdirSync, existsSync, realpathSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = new URL("./", import.meta.url);
const LIB = new URL("../", HERE), RESEARCH = new URL("../../research/", HERE);
const NETS_SRC = new URL("../core/nets/", HERE);
const { readNet } = await import(new URL("gpu/cnn/netfile.mjs", LIB).href);
const { readQuantC, forwardCodesC, readReferenceCQ81 } = await import(new URL("gpu/cnn/q8c.mjs", LIB).href);
const { requant, dequant } = await import(new URL("gpu/cnn/q8.mjs", LIB).href);

// ai: The net: its q8 file (the installed int8 twin, liblizard/gpu/wgsl/classify.mjs INT8_TWINS), the setup's pipeline
// ai: and buffer it replaces, and each conv's GEMM kind.
export const NETS = {
  v3: { file: "gpu/cnn/weights-v3-q8.safetensors", ref: "build/cnn/reference-v3-q8.bin", arch: "cnn-v3", src: "cls_v3.comp", role: "front.twins.small.pipe", weights: "front.twins.small.buf", kinds: ["c1", "s2c8", "s2t2"] },
};
// ai: The kernel's specialisation constants, in constant_id order (core/nets/cls.glsl): M (coopmat rows), NW (the wide
// ai: N tile), FAST (1 the fast build: a lane builds and consumes its own GEMM row in registers where the device's
// ai: element layout is lane = row, checked by the kernel at run time; 0 the staging build, through shared memory, on
// ai: any layout), STOP (0: the whole net; a profiling build stops after layer STOP),
// ai: DBG (1: each reading records the path its workgroup took and the stride unit probed), PP (patches a workgroup,
// ai: 1 to 4). A check edits a manifest's spec array, no rebuild. su: the kernel's -DSU, what a coopMatLoad stride
// ai: counts for an int8 matrix in u32 memory on the device (1 words, as SPIR-V says; 4 bytes, the Adreno 840's
// ai: driver); a build define because the Adreno is fast only where a stride is a literal.
const SPEC = ["M", "NW", "FAST", "STOP", "DBG", "PP"];
export const SHAPES = {
  adreno: { spec: { M: 64, NW: 32, FAST: 1, STOP: 0, DBG: 0, PP: 4 }, su: 4, needs: [{ M: 64, N: 32, K: 32 }, { M: 64, N: 16, K: 32 }] },
  nv16: { spec: { M: 16, NW: 16, FAST: 0, STOP: 0, DBG: 0, PP: 4 }, su: 1, needs: [{ M: 16, N: 16, K: 32 }] },
};
const I8 = 3, I32 = 5;   // ai: VkComponentTypeKHR

const STEPS = { c1: 1, s2c8: 3, s2t2: 5 };
const colsOf = (kind, L) => (kind === "c1" ? 32 : L.cout);
const chanOf = (kind, L, n) => (kind === "c1" ? n % 8 : n);

// ai: B_L's byte (s, k, n): the weight code the GEMM multiplies there, or 0.
function weightAt(kind, L, s, k, n) {
  const w = (co, c, ky, kx) => L.wq[co * L.k + c * 9 + ky * 3 + kx];
  if (kind === "c1") {
    const j = n >> 3, co = n & 7;
    for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
      const e = 2 * j + kx - 1;
      if ((e >= 0 ? 8 * ky + e : 24 + ky) === k) return w(co, 0, ky, kx);
    }
    return 0;
  }
  if (kind === "s2c8") { if (k >= 24) return 0; return w(n, k & 7, s, k >> 3); }
  if (kind === "s2t2") { const t = 2 * s + (k >> 4); return t <= 8 ? w(n, k & 15, Math.floor(t / 3), t % 3) : 0; }
  throw new Error(kind);
}

// ai: The weights buffer of one net and the offsets (u32 words) the kernel is compiled with.
export function pack(q, net) {
  const convs = q.layers.filter((L) => L.type === "conv"), fc1 = q.layers.find((L) => L.name === "fc1");
  if (convs.length !== net.kinds.length) throw new Error(`${net.arch}: ${convs.length} convs, the kernel has ${net.kinds.length}`);
  const parts = [], defs = {};
  let at = 0;
  const put = (name, bytes) => { if (bytes.length % 16) throw new Error(`${name}: ${bytes.length} bytes`); defs[name] = at; parts.push(bytes); at += bytes.length / 4; };
  const i32s = (a) => new Uint8Array(Int32Array.from(a).buffer);
  const f32s = (a) => new Uint8Array(Float32Array.from(a).buffer);
  const pad4 = (a, v = 0) => { const o = Array.from(a); while (o.length % 4) o.push(v); return o; };
  convs.forEach((L, i) => {
    const kind = net.kinds[i], S = STEPS[kind], N = colsOf(kind, L), B = new Int8Array(S * 32 * N);
    for (let s = 0; s < S; s++) for (let k = 0; k < 32; k++) for (let n = 0; n < N; n++) B[(s * 32 + k) * N + n] = weightAt(kind, L, s, k, n);
    // ai: every nonzero weight placed once per column: a check that no tap was lost or doubled
    let placed = 0;
    for (let n = 0; n < N; n++) for (let s = 0; s < S; s++) for (let k = 0; k < 32; k++) placed += B[(s * 32 + k) * N + n] !== 0;
    let want = 0;
    for (let n = 0; n < N; n++) { const co = chanOf(kind, L, n); for (let j = 0; j < L.k; j++) want += L.wq[co * L.k + j] !== 0; }
    if (placed !== want) throw new Error(`${net.arch} ${L.name}: ${placed} weights placed of ${want}`);
    put(`OB${i + 1}`, new Uint8Array(B.buffer));
    put(`OBQ${i + 1}`, i32s(Array.from({ length: N }, (_, n) => L.bq[chanOf(kind, L, n)])));
    put(`OM${i + 1}`, f32s(Array.from({ length: N }, (_, n) => L.mul[chanOf(kind, L, n)])));
  });
  const last = convs[convs.length - 1], C = last.cout, K1 = fc1.k, side = last.sideOut;
  if (fc1.cout !== 32 || side !== 4 || K1 !== C * 16 || K1 % 32) throw new Error(`${net.arch}: fc1 ${fc1.cin} x ${side}^2 -> ${fc1.cout}`);
  const A1 = new Int8Array(32 * K1), H = K1 / 2;
  for (let o = 0; o < 32; o++) for (let pos = 0; pos < 16; pos++) for (let c = 0; c < C; c++) {
    const k = pos * C + c, h = k >= H ? 1 : 0;
    A1[(h * 32 + o) * H + (k - h * H)] = fc1.wq[o * K1 + c * 16 + pos];
  }
  put("OAF1", new Uint8Array(A1.buffer));
  put("OBQF1", i32s(fc1.bq));
  put("OSF1", f32s(fc1.mul));
  const { cin, w, b } = q.fc2;
  if (cin !== 32) throw new Error(`${net.arch}: fc2 over ${cin}`);
  put("OWF2", f32s(Array.from(w, (v) => Math.fround(v))));
  put("OBF2", f32s(pad4(Array.from(b, (v) => Math.fround(v)))));
  const bytes = new Uint8Array(at * 4);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  return { bytes, defs, K1, S1: K1 / 64 };
}

// ai: ---- The kernel's GEMM formulation on the CPU (core/nets/cls.glsl's row builders, byte for byte) ----

// ai: The 32 bytes of GEMM row r at step s (signed codes), from the layer's input map (NHWC, a patch's; conv1 reads
// ai: Xq as [y][x]); p the patch within the 4 for the kind whose rows span 4 patches (s2t2, conv3), whose maps are
// ai: then [patch][...].
function buildRow(kind, L, map, r, s, spans4) {
  const row = new Int32Array(32), S = L.side, C = L.cin;
  const at = (p, y, x, c) => (y < 0 || y >= S || x < 0 || x >= S ? 0 : map[((p * S + y) * S + x) * C + c]);
  if (kind === "c1") {
    const y = r >> 2, xg = r & 3;
    for (let ky = 0; ky < 3; ky++) {
      const iy = 2 * y + ky - 1;
      for (let c = 0; c < 8; c++) row[8 * ky + c] = at(0, iy, 8 * xg + c, 0);
      row[24 + ky] = at(0, iy, 8 * xg - 1, 0);
    }
  } else if (kind === "s2c8") {
    const y = r >> 3, x = r & 7, iy = 2 * y + s - 1;
    for (let kx = 0; kx < 3; kx++) for (let c = 0; c < 8; c++) row[8 * kx + c] = at(0, iy, 2 * x - 1 + kx, c);
  } else if (kind === "s2t2") {
    const so = L.sideOut, p = spans4 ? r >> 4 : 0, rr = spans4 ? r & 15 : r, y = Math.floor(rr / so), x = rr % so;
    for (let h = 0; h < 2; h++) {
      const t = 2 * s + h;
      if (t > 8) continue;
      for (let c = 0; c < 16; c++) row[16 * h + c] = at(p, 2 * y + Math.floor(t / 3) - 1, 2 * x + (t % 3) - 1, c);
    }
  }
  return row;
}

// ai: The net on 4 patches' Xq (Int8Array(1024) each) through the packed buffer, as the kernel runs it: the codes of
// ai: every conv (NHWC, per patch), fc1's outputs and the five outputs in the kernel's f32 order.
export function emulate(q, net, blob, xqs) {
  const u8 = blob.bytes, i8 = new Int8Array(u8.buffer, u8.byteOffset, u8.length), dv = new DataView(u8.buffer, u8.byteOffset, u8.length);
  const I = (w) => dv.getInt32(4 * w, true), F = (w) => dv.getFloat32(4 * w, true);
  const convs = q.layers.filter((L) => L.type === "conv"), P = xqs.length;
  let maps = xqs.map((x) => Int8Array.from(x));
  const codes = [];
  convs.forEach((L, i) => {
    const kind = net.kinds[i], S = STEPS[kind], N = colsOf(kind, L), OB = blob.defs[`OB${i + 1}`], OBQ = blob.defs[`OBQ${i + 1}`], OM = blob.defs[`OM${i + 1}`];
    const spans4 = kind === "s2t2" && L.sideOut === 4;
    const rows = kind === "c1" ? 64 : L.sideOut * L.sideOut;
    const outBytes = L.sideOut * L.sideOut * L.cout;
    const run = (map, R, pOut) => {
      const out = new Int8Array(pOut * outBytes);
      for (let r = 0; r < R; r++) {
        const acc = new Array(N).fill(0);
        for (let s = 0; s < S; s++) {
          const a = buildRow(kind, L, map, r, s, spans4);
          for (let n = 0; n < N; n++) { let t = 0; for (let k = 0; k < 32; k++) t += a[k] * i8[4 * OB + (s * 32 + k) * N + n]; acc[n] += t; }
        }
        for (let n = 0; n < N; n++) out[r * N + n] = requant(acc[n], I(OBQ + n), F(OM + n), q.qmax);
      }
      return out;
    };
    if (spans4) {
      const all = new Int8Array(P * L.side * L.side * L.cin);
      maps.forEach((m, p) => all.set(m, p * m.length));
      const out = run(all, 64, P);
      maps = Array.from({ length: P }, (_, p) => out.subarray(p * outBytes, (p + 1) * outBytes));
    } else maps = maps.map((m) => run(m, rows, 1));
    codes.push(maps);
  });
  const K1 = blob.K1, H = K1 / 2, OA = blob.defs.OAF1, a = [], out = [];
  for (let p = 0; p < P; p++) {
    const ap = new Float32Array(32);
    for (let o = 0; o < 32; o++) {
      // ai: the two diagonal blocks: row o over the first half, row 32 + o over the second
      let acc = 0;
      for (let h = 0; h < 2; h++) for (let k = 0; k < H; k++) acc += i8[4 * OA + (h * 32 + o) * H + k] * maps[p][h * H + k];
      ap[o] = dequant(acc, I(blob.defs.OBQF1 + o), F(blob.defs.OSF1 + o));
    }
    a.push(ap);
    const z = new Float32Array(5);
    for (let o = 0; o < 5; o++) {
      let acc = F(blob.defs.OBF2 + o);
      for (let k = 0; k < 32; k++) acc = Math.fround(acc + Math.fround(F(blob.defs.OWF2 + o * 32 + k) * ap[k]));
      z[o] = acc;
    }
    out.push(z);
  }
  return { codes, a, out };
}

// ai: The CPU check: the trainer's CQ81 reference patches (their Xq; the file's first `real` patches, where research/
// ai: has the file: it is not published, and the line says when it is not there) and random ones (uniform codes, and
// ai: sparse ones of the extreme codes) through emulate() against forwardCodesC, every conv code exact, fc1 exact (the
// ai: same f32 op on the same integer), the outputs within 1e-5 of the contract's f64 sum over the sum of the terms'
// ai: magnitudes.
export function check(q, net, blob, rounds = 16, real = 256) {
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
  const convs = q.layers.filter((L) => L.type === "conv");
  let codes = 0, bad = 0, worst = 0, fbad = 0;
  const refFile = new URL(net.ref, RESEARCH), ref = existsSync(refFile) ? readReferenceCQ81(readFileSync(refFile)) : null;
  const R = ref ? Math.min(real, ref.count) >> 2 : 0;
  for (let round = 0; round < rounds + R; round++) {
    const xqs = Array.from({ length: 4 }, (_, p) => round < R ? Int8Array.from(ref.xq.subarray((4 * round + p) * 1024, (4 * round + p + 1) * 1024)) : Int8Array.from({ length: 1024 }, () => {
      const v = rnd();
      return (round + p) % 3 === 0 ? (v < 0.2 ? -127 : v > 0.8 ? 127 : 0) : Math.round(rnd() * 254) - 127;
    }));
    const em = emulate(q, net, blob, xqs);
    for (let p = 0; p < 4; p++) {
      const ref = forwardCodesC(q, xqs[p]);
      convs.forEach((L, i) => {
        const S = L.sideOut, Cn = L.cout, got = em.codes[i][p];
        for (let c = 0; c < Cn; c++) for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
          codes++;
          if (got[(y * S + x) * Cn + c] !== ref.codes[i][(c * S + y) * S + x]) bad++;
        }
      });
      for (let o = 0; o < 32; o++) if (em.a[p][o] !== ref.a[o]) fbad++;
      for (let o = 0; o < 5; o++) {
        let mag = Math.abs(q.fc2.b[o]);
        for (let c = 0; c < 32; c++) mag += Math.abs(q.fc2.w[o * 32 + c] * ref.a[c]);
        worst = Math.max(worst, Math.abs(em.out[p][o] - ref.out[o]) / Math.max(1, mag));
      }
    }
  }
  const ok = bad === 0 && fbad === 0 && worst < 1e-5;
  return { ok, line: `${net.arch}: ${ref ? `${4 * R} reference` : `0 reference (no research/${net.ref}: the trainer's, not published)`} and ${rounds * 4} random patches, ${bad} of ${codes} codes differ, fc1 ${fbad} differ, outputs within ${worst.toExponential(2)} ${ok ? "PASS" : "FAIL"}` };
}

const readQ = (net) => readQuantC(readNet(readFileSync(new URL(net.file, LIB))));

// ai: glslang: the kernel's SPIR-V (Vulkan 1.3), validated by the SPIRV-Tools glslang links (--spirv-val).
function compile(glslang, net, defs, outPath) {
  const D = Object.entries(defs).map(([k, v]) => `-D${k}=${v}`);
  execFileSync(glslang, ["-V", "--target-env", "vulkan1.3", "--spirv-val", "-g0", ...D, "-o", outPath, fileURLToPath(new URL(net.src, NETS_SRC))], { stdio: ["ignore", "pipe", "inherit"] });
}

// ai: The shortest decimal literal that gives back f32 v in GLSL.
function f32Literal(v) {
  for (let p = 6; p <= 9; p++) { const s = (+v).toPrecision(p); if (Math.fround(+s) === Math.fround(v)) return /[.e]/.test(s) ? s : `${s}.0`; }
  throw new Error(`${v}: no f32 literal`);
}

export async function build({ out, glslang }) {
  if (process.env.LIZ_NETS_CLS !== "1") {
    console.log("  cnn-v3: no kernel in the manifest (off by default: 1.5 to 1.8 s more cold start, core/nets/README.md; LIZ_NETS_CLS=1 builds it)");
    return [];
  }
  const shapes = SHAPES[process.env.LIZ_NETS_SHAPES ?? "adreno"];
  if (!shapes) throw new Error(`LIZ_NETS_SHAPES=${process.env.LIZ_NETS_SHAPES}: adreno or nv16`);
  mkdirSync(new URL("nets/", out), { recursive: true });
  const entries = [];
  for (const [name, net] of Object.entries(NETS)) {
    const q = readQ(net), blob = pack(q, net), c = check(q, net, blob);
    console.log(`  ${c.line}`);
    if (!c.ok) throw new Error(`${net.arch}: the packed GEMM is not the contract`);
    if (q.qmax !== 127) throw new Error(`${net.arch}: ${q.bits} bits; the kernel clamps at 127`);
    const spv = `nets/cls_${name}.spv`, bin = `nets/cls_${name}.bin`;
    writeFileSync(new URL(bin, out), blob.bytes);
    compile(glslang, net, { ...blob.defs, INV: f32Literal(q.inv), SF1: blob.S1, K1: blob.K1, SU: `${shapes.su}u` }, fileURLToPath(new URL(spv, out)));
    const spec = SPEC.map((k) => shapes.spec[k]);
    entries.push({ variants: ["int8-sg", "int8"], role: net.role, weights: net.weights, spv, spec, P: shapes.spec.PP, blob: bin, needs: shapes.needs.map((s) => ({ ...s, A: I8, B: I8, C: I32, R: I32 })) });
    console.log(`  ${net.arch}: ${bin} ${blob.bytes.length} B, ${spv}, spec ${JSON.stringify(spec)}`);
  }
  return entries;
}

// ai: Two `lizard_gpu_check dump` directories (the web's twins against these kernels, or two runs of either) held per
// ai: peak: SELECT's slot order is atomic, so a peak's slot moves between runs, and its set can differ at the cap. Each
// ai: slot of the first F frames is keyed by its peak's four words; its readings (12 words) and the small net's logit
// ai: are compared by key, split by the large net's list: a peak in both lists holds the large net's readings, in
// ai: neither the small net's, in one list only (RANK saw another set) neither pair. Returns the counts and the
// ai: differing readings.
export function compareDumps(A, Bd, { F = 8, cap = 256, keep = 64 } = {}) {
  const load = (d) => { const u = (n) => { const b = readFileSync(`${d}/${n}.bin`); return new Uint32Array(b.buffer, b.byteOffset, b.length / 4); }; return { peaks: u("peaks"), readings: u("readings"), logit: u("logit"), list: u("list"), counts: u("counts") }; };
  const a = load(A), b = load(Bd), f32 = (w) => new Float32Array(Uint32Array.of(w).buffer)[0];
  const r = { peaks: 0, missing: 0, large: [0, 0], small: [0, 0], moved: [0, 0], logits: 0, diffs: [] };
  for (let f = 0; f < F; f++) {
    const n = Math.min(a.counts[f * 16 + 2], cap), key = (P, i) => Array.from(P.peaks.subarray(4 * (f * cap + i), 4 * (f * cap + i) + 4)).join(",");
    const at = new Map();
    for (let i = 0; i < Math.min(b.counts[f * 16 + 2], cap); i++) at.set(key(b, i), i);
    const listed = (P) => new Set(P.list.subarray(f * keep, f * keep + Math.min(P.counts[f * 16 + 3], keep)));
    const la = listed(a), lb = listed(b);
    for (let i = 0; i < n; i++) {
      r.peaks++;
      const j = at.get(key(a, i));
      if (j === undefined) { r.missing++; continue; }
      const cls = la.has(i) && lb.has(j) ? "large" : !la.has(i) && !lb.has(j) ? "small" : "moved";
      const ra = a.readings.subarray(12 * (f * cap + i), 12 * (f * cap + i) + 12), rb = b.readings.subarray(12 * (f * cap + j), 12 * (f * cap + j) + 12);
      const bad = ra.some((w, k) => w !== rb[k]);
      r[cls][0]++; r[cls][1] += bad;
      if (bad && cls !== "moved") r.diffs.push({ f, cls, peak: [f32(ra[0]), f32(ra[1])], a: Array.from(ra, f32), b: Array.from(rb, f32) });
      if (a.logit[f * cap + i] !== b.logit[f * cap + j]) r.logits++;
    }
  }
  return r;
}

// ai: Run as a script (not when gen/nets.mjs imports build), by real path: through a symlink or a space import.meta.url
// ai: and argv[1] spell the one file two ways.
if ((() => { try { return fileURLToPath(import.meta.url) === realpathSync(process.argv[1]); } catch { return false; } })()) {
  const cmd = process.argv[2];
  if (cmd === "check") {
    let ok = true;
    for (const net of Object.values(NETS)) { const q = readQ(net), c = check(q, net, pack(q, net), 64); console.log(c.line); ok = ok && c.ok; }
    process.exit(ok ? 0 : 1);
  } else if (cmd === "compare" && process.argv.length >= 5) {
    const r = compareDumps(process.argv[3], process.argv[4], { F: +(process.argv[5] ?? 8) });
    console.log(`${r.peaks} peaks, ${r.missing} not in both runs; readings differ: the large net's ${r.large[1]} of ${r.large[0]}, the small net's ${r.small[1]} of ${r.small[0]} (and ${r.moved[1]} of ${r.moved[0]} listed in one run only); logits differ on ${r.logits}`);
    for (const d of r.diffs) console.log(`  frame ${d.f} ${d.cls} peak (${d.peak.join(", ")}): ${d.a.slice(2, 5).map((v) => v.toPrecision(7)).join(" ")} (form ${d.a[8]}) against ${d.b.slice(2, 5).map((v) => v.toPrecision(7)).join(" ")} (form ${d.b[8]})`);
    process.exit(r.large[1] || r.small[1] || r.logits ? 1 : 0);
  } else {
    console.error("usage: node liblizard/gen/nets_cls.mjs check | compare <dump dir A> <dump dir B> [frames]   (the build runs from gen/nets.mjs)");
    process.exit(2);
  }
}
