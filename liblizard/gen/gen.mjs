// ai: The Android decoder's setup, taken from the web decoder itself: the web host (liblizard/gpu/decoder.mjs
// ai: FrontHalf.create, with every option the receiver's lizard-web/gpuqueue.mjs openGpu names) is built on a recording
// ai: device (mockgpu.mjs) once a variant, and what it made is written out: every shader module's WGSL (rewritten where
// ai: naga reads WGSL otherwise than Tint, `nagaWgsl`) compiled to SPIR-V by naga, every bind group layout and
// ai: pipeline, every static buffer with the bytes the host wrote into it, and the values the lanes and batches are
// ai: made from (`exportTree`). The native host (core/dec/) builds exactly these objects and ports only what a lane and
// ai: a batch do. Nothing here decodes; the tables come from liblizard/build/ob.wasm as the web's do.
//
// ai: A variant is (precision, subgroups): precision the web's precisionChain choice (int8: the int8 proposer, the
// ai: classifiers f16 with their int8 twins built beside; f16; f32), subgroups whether the classifiers reduce by
// ai: shuffles. The back half is built at every batch size a device may need (B_CEIL down to 1: plan() rebuilds it
// ai: at Bmax where memory holds fewer frames), all at workgroup memory 32 KB, the floor every phone we know has.
//
//   node liblizard/gen/gen.mjs [out]      (lizard-android/build.sh gen; out defaults to liblizard/out)
// ai: Writes out/setup/<variant>.json, out/blobs/<hash>.bin (the buffers' bytes), out/wgsl/<hash>.wgsl (as the web host made it),
// ai: out/naga/<hash>.wgsl (as naga reads it) and out/spv/<hash>.spv, and out/gen.json (the variants, the modules, the
// ai: tools' versions and the gpu sources' hash that made them). Needs, each checked before anything runs: ../build/ob.mjs
// ai: and ob.wasm (liblizard's ./build.sh); wgsl2spv and naga (lizard-android/build.sh tools); SPIRV-Tools' spirv-val (and
// ai: spirv-opt under LIZ_SPIRV_OPT=1) and shaderc's glslc, the Android NDK's or else the PATH's.
import { installGlobals, Recorder, mockAdapter, hash } from "./mockgpu.mjs";
import { senderSetup } from "./sender.mjs";
import { mkdirSync, writeFileSync, readFileSync, existsSync, readdirSync, statSync, realpathSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { homedir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

installGlobals();
const HERE = new URL("./", import.meta.url), LIB = new URL("../", import.meta.url);
const OUT = process.argv[2] ? pathToFileURL(`${resolve(process.argv[2])}/`) : new URL("../out/", HERE);
const imp = (p) => import(new URL(p, LIB).href);
// ai: Run as a script (not when a check imports stripCaps or nagaWgsl), by real path: through a symlink or a space
// ai: import.meta.url and argv[1] spell the one file two ways.
const MAIN = (() => { try { return fileURLToPath(import.meta.url) === realpathSync(process.argv[1]); } catch { return false; } })();

const NDK = process.env.ANDROID_NDK_HOME || `${process.env.ANDROID_HOME || `${homedir()}/Android/Sdk`}/ndk/27.1.12297006`;
// ai: SPIRV-Tools' spirv-val (and spirv-opt under LIZ_SPIRV_OPT) and shaderc's glslc: the Android NDK's shader-tools
// ai: where there is an NDK, else the system's on the PATH
const EXE = process.platform === "win32" ? ".exe" : "";
const shaderTool = (name) => {
  const ndk = join(NDK, "shader-tools", { darwin: "darwin-x86_64", win32: "windows-x86_64" }[process.platform] ?? "linux-x86_64", name + EXE);
  if (existsSync(ndk)) return ndk;
  for (const dir of (process.env.PATH ?? "").split(delimiter)) if (dir && existsSync(join(dir, name + EXE))) return join(dir, name + EXE);
  return ndk;
};
// ai: wgsl2spv (gen/wgsl2spv, naga 30 as a library: lizard-android/build.sh tools builds it) in place of naga-cli, whose
// ai: writer zeroes workgroup memory from one invocation and bounds every loop (wgsl2spv/src/main.rs says what that
// ai: cost); naga-cli (../.tools/bin/naga) only for its version line.
export const TOOLS = { wgsl2spv: fileURLToPath(new URL(`./wgsl2spv/target/release/wgsl2spv${EXE}`, HERE)), naga: fileURLToPath(new URL(`../.tools/bin/naga${EXE}`, HERE)), val: shaderTool("spirv-val"), opt: shaderTool("spirv-opt"), glslc: shaderTool("glslc") };
// ai: spirv-opt -O on the SPIR-V: off unless LIZ_SPIRV_OPT=1. Tint's SPIR-V reaches the driver as written, and -O
// ai: unrolled the fused pass 1 and F8's word to three times their size (2026-09-29).
const OPT = process.env.LIZ_SPIRV_OPT === "1";

// ai: What a run needs, checked before the gpu modules load (sim/ob.mjs imports ../build/ob.mjs): every one missing,
// ai: each with where it comes from.
if (MAIN) {
  const ndk = (or) => `the Android NDK's shader-tools: ANDROID_NDK_HOME; or ${or} on the PATH`;
  const missing = [
    [fileURLToPath(new URL("build/ob.mjs", LIB)), "liblizard/build.sh"], [fileURLToPath(new URL("build/ob.wasm", LIB)), "liblizard/build.sh"],
    [TOOLS.wgsl2spv, "lizard-android/build.sh tools"], [TOOLS.naga, "lizard-android/build.sh tools"],
    [TOOLS.val, ndk("SPIRV-Tools")], ...(OPT ? [[TOOLS.opt, ndk("SPIRV-Tools")]] : []), [TOOLS.glslc, ndk("shaderc")],
  ].filter(([p]) => !existsSync(p));
  if (missing.length) { console.error(missing.map(([p, from]) => `no ${p} (${from})`).join("\n")); process.exit(1); }
}

const { FrontHalf, CAP, CAP_MS, B_CEIL, LEVELS, COUNTERS, TAU, EPS, TAU_TRACK } = await imp("gpu/decoder.mjs");
const { formatTables, PICTURES } = await imp("gpu/tables.mjs");
const { loadNet } = await imp("gpu/cnn/netfile.mjs");
const { CASCADE } = await imp("gpu/wgsl/classify.mjs");
const { K, CELL, HYP, RING_COUNT } = await imp("gpu/wgsl/finder.mjs");
const { NODES_MAX, REFIT_DBG } = await imp("gpu/wgsl/register.mjs");
const { HIST_BINS, PYRAMID_BLOCK } = await imp("gpu/wgsl/pyramid.mjs");
const { KEEP } = await imp("gpu/wgsl/bank.mjs");
const { TILE } = await imp("gpu/wgsl/bank_fcn2.mjs");
const { SLOTS } = await imp("gpu/wgsl/common.mjs");
const { ARGS_WORDS, ARGS_SLOT, argsLdpc, argsWords } = await imp("gpu/wgsl/back_transform.mjs");
const { VERSION_MAX } = await imp("gpu/wordcode.mjs");
const { COUNTS } = await imp("gpu/back/back.mjs");
const { REC_BYTES } = await imp("gpu/back/ldpc.mjs");
const { PAYLOAD } = await imp("gpu/back/ref_ldpc.mjs");

// ai: The receiver's classifiers (lizard-web/gpuqueue.mjs INSTALLED): cnn-v1, and cnn-v3 as the cascade's first net
// ai: (CASCADE.weights is a path under liblizard/ since 2026-10-01, as the app's build needs it).
const LARGE = "gpu/cnn/weights.safetensors", SMALL = CASCADE.weights;
// ai: The device the setup is made for: 32 KB of workgroup memory (every phone we know, and what Chrome gives the
// ai: desktop's GPUs), the rest generous; plan()'s memory limits are the real device's, read at run time.
const LIMITS = { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31, maxComputeWorkgroupStorageSize: 32768, maxStorageBuffersPerShaderStage: 16, maxTextureArrayLayers: 2048, maxComputeInvocationsPerWorkgroup: 256 };
const FEATURES = ["timestamp-query", "shader-f16", "subgroups"];
export const VARIANTS = ["int8", "f16", "f32"].flatMap((precision) => [true, false].map((subgroups) => ({ precision, subgroups, name: `${precision}${subgroups ? "-sg" : ""}` })));
const BACK_B = [B_CEIL, 16, 8, 4, 2, 1];
// ai: The ingest shader (F0, a render pass over a texture_external): the native app writes luma into the layer
// ai: itself, so it is neither exported nor compiled.
const SKIP = (code) => /texture_external/.test(code);

// ai: The WGSL as naga 30 reads it: Tint takes three constructs naga does not, each rewritten here to the same
// ai: arithmetic:
// ai:   `enable subgroups;`: naga has the subgroup builtins without it and refuses the directive;
// ai:   a bitcast between f16 vectors and u32s (bitcast<vec4<f16>>(vec2<u32>), bitcast<vec2<f16>>(u32) and back):
// ai:     naga has none; the halves are unpacked to f32 and converted back (pack2x16float the other way), which is
// ai:     exact for every f16 but a denormal a device flushes. A cast to u32s is rewritten only where its argument
// ai:     names f16 (the same spelling casts f32 vectors elsewhere, which naga takes).
const HELPERS = {
  liz_h4: "fn liz_h4(w: vec2<u32>) -> vec4<f16> { return vec4<f16>(vec2<f16>(unpack2x16float(w.x)), vec2<f16>(unpack2x16float(w.y))); }",
  liz_h2: "fn liz_h2(w: u32) -> vec2<f16> { return vec2<f16>(unpack2x16float(w)); }",
  liz_p4: "fn liz_p4(h: vec4<f16>) -> vec2<u32> { return vec2<u32>(pack2x16float(vec2f(h.xy)), pack2x16float(vec2f(h.zw))); }",
  liz_p2: "fn liz_p2(h: vec2<f16>) -> u32 { return pack2x16float(vec2f(h)); }",
  liz_pack4xI8: "fn liz_pack4xI8(v: vec4<i32>) -> u32 { let u = bitcast<vec4<u32>>(v) & vec4<u32>(0xffu); return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u); }",
};
// ai:   pack4xI8: naga lowers it to a conversion into 8-bit integers and a bitcast of that vector to u32, which the
// ai:     S26's Adreno 840 gets wrong (the odd bytes of the proposer's int8 plane came back 0, and the int8 variant
// ai:     found nothing, 2026-09-29); rewritten to the low bytes shifted together, WGSL's definition (Tint's form).
const CASTS = [["pack4xI8(", "liz_pack4xI8", false], ["bitcast<vec4<f16>>(", "liz_h4", false], ["bitcast<vec2<f16>>(", "liz_h2", false], ["bitcast<vec2<u32>>(", "liz_p4", true], ["bitcast<u32>(", "liz_p2", true]];
// ai: The argument of the call whose "(" ends at i - 1: up to its balancing ")".
function argAt(s, i) {
  let depth = 1, j = i;
  for (; j < s.length && depth; j++) { if (s[j] === "(") depth++; else if (s[j] === ")") depth--; }
  return s.slice(i, j - 1);
}
export function nagaWgsl(code) {
  let s = code.replace(/^\s*enable subgroups;\s*$/m, "");
  const used = new Set();
  for (const [from, to, onlyF16] of CASTS) {
    let out = "", at = 0;
    for (let i = s.indexOf(from); i >= 0; i = s.indexOf(from, at)) {
      const arg = argAt(s, i + from.length);
      if (onlyF16 && !arg.includes("f16")) { out += s.slice(at, i + from.length); at = i + from.length; continue; }
      out += s.slice(at, i) + `${to}(`; at = i + from.length; used.add(to);
    }
    s = out + s.slice(at);
  }
  if (used.size) {
    // ai: After the directives, which WGSL wants before any declaration.
    const lines = s.split("\n");
    let at = 0;
    while (at < lines.length && /^\s*(enable|requires|diagnostic)\b|^\s*$|^\s*\/\//.test(lines[at])) at++;
    lines.splice(at, 0, ...[...used].map((k) => HELPERS[k]));
    s = lines.join("\n");
  }
  return s;
}

// ai: A buffer, texture view or pipeline of the recorder as its id; everything else as it is.
const idOf = (o) => (o == null ? null : o.__id ?? null);
const pb = (x) => (x ? { p: idOf(x.p), bgl: idOf(x.bgl) } : null);
// ai: A classifier's pipe: P, the patches a workgroup of its kernel takes (its dispatch is ceil(slots / P)).
const pbNet = (x) => (x ? { ...pb(x), P: x.P } : null);
const perSlot = (arr) => Array.from({ length: SLOTS }, (_, s) => idOf(arr?.[s]));
const plain = (v) => JSON.parse(JSON.stringify(v, (k, x) => (ArrayBuffer.isView(x) ? Array.from(x) : x)));

// ai: What the lanes and batches read of the back half (gpu/back/back.mjs, transform.mjs, soft.mjs, ldpc.mjs).
function exportBack(bh) {
  const t = bh.transform, so = bh.soft, ld = bh.ldpc;
  return {
    B: bh.B, precision: bh.precision, dims: plain(bh.dims), bytes: bh.bytes, readBytes: bh.readBytes,
    transform: {
      built: t.built, zero: t.zero, cbytes: t.cbytes, yStride: t.yStride, sStride: t.sStride, partStride: t.partStride, sampStride: t.sampStride, nmax: t.nmax,
      gate: idOf(t.gate), gateBgl: idOf(t.gateBgl), p1fBgl: idOf(t.p1fBgl), redBgl: idOf(t.redBgl), p2Bgl: idOf(t.p2Bgl),
      pass1f: perSlot(t.pass1f), reduce: perSlot(t.reduce), pass2: perSlot(t.pass2),
      twBufs: perSlot(t.twBufs), x12Bufs: perSlot(t.x12Bufs), rowsBufs: perSlot(t.rowsBufs), picUnis: perSlot(t.picUnis), gateUni: idOf(t.gateUni),
      sizes: t.tables.map((tb) => (tb ? { n: tb.n, subch: tb.subch, blocks: tb.blocks, m2: tb.m2, sx2: tb.sx2, sq2: tb.sq2 } : null)),
    },
    soft: { pipeline: idOf(so.pipeline), align: idOf(so.alignPipeline), bgl: idOf(so.bgl), uvBuf: idOf(so.uvBuf), twBuf: idOf(so.twBuf), params: perSlot(so.params), served: so.served, blocks: so.sizes.map((z) => (z ? z.blocks : 0)), bytes: so.bytes, indirect: so.indirect },
    ldpc: { pipeline: idOf(ld.pipeline), bgl: idOf(ld.bgl), mapBuf: idOf(ld.mapBuf), params: idOf(ld.params), bytes: ld.bytes, blocksMax: ld.blocksMax, recCap: ld.recCap },
  };
}

// ai: What the lanes and batches read of the front half (gpu/decoder.mjs lane, ensure, batch; gpu/bank_fcn2.mjs
// ai: ensure, dispatch), with the host's constants, so the native code restates none of them.
function exportTree(fh, v, tables) {
  const tf = fh.twinForms;
  const twin = (t) => (t ? { pipe: pbNet(t.pipe), buf: idOf(t.buf), precision: t.w.precision } : null);
  const f = fh.fcn2Form;
  return {
    variant: v.name, precision: fh.precision, floatPrecision: fh.floatPrecision, subgroups: v.subgroups, features: fh.features, nets: fh.nets,
    consts: { LEVELS, COUNTERS, RESULT: 32, SEL_WORDS: 4, TS_MAX: 80, TS_PLAIN: 2, TS_INGEST: 2, RING_SLACK: 4, K, CELL, HYP, RING_COUNT, NODES_MAX, REFIT_DBG, HIST_BINS, PYRAMID_BLOCK, TAU, EPS, TAU_TRACK,
      TILE, KEEP, B_CEIL, CAP_MS, TWIN_FRAMES: 2, TWIN_ROUNDS: 3, VERSION_MAX, SLOTS, ARGS_WORDS, ARGS_SLOT, argsLdpc, argsWords, COUNTS, REC_BYTES, PAYLOAD },
    cap: fh.cap, register: fh.register, nodesMax: fh.nodesMax, modules: Array.from(fh.modules), bankKeep: fh.bank.KEEP,
    cascade: fh.cascade ? { keep: fh.cascade.keep, rest: fh.cascade.rest, precision: fh.cascade.weights.precision } : null, weightsPrecision: fh.weights.precision,
    front: {
      pyramid: pb(fh.pyramid), classify: pbNet(fh.classify), classify0: pbNet(fh.classify0), rank: pb(fh.rank),
      vote: pb(fh.vote), gather: pb(fh.gather), score: pb(fh.score), pick: pb(fh.pick), nodesP: pb(fh.nodesP), refit: pb(fh.refit), wordP: pb(fh.wordP), select: pb(fh.select),
      fcn2: pb(fh.fcn2), fcn2X: pb(fh.fcn2X), fcn2Form: { TX: f.TX, TY: f.TY, XB: f.XB, walk: f.walk, xplane: f.xplane, int8: f.int8, store: f.store, name: f.name },
      buffers: Object.fromEntries(["sizesBuf", "planBuf", "planUni", "p4Uni", "pickUni", "nodesBuf", "tmplBuf", "latticeBuf", "latUni", "wordPtsBuf", "wordUni", "wordLatUni", "fitUni", "weightsBuf", "weightsBuf0"].map((k) => [k, idOf(fh[k])])),
      rounds: fh.rounds.map(idOf),
      twins: tf ? { large: twin(tf.large), small: twin(tf.small) } : null,
    },
    pictures: PICTURES.map(({ n, subch, lo, hi }) => ({ n, subch, lo, hi })),
    rings: tables.rings.map((t) => ({ nodeX: Array.from(t.nodeX), nodeY: Array.from(t.nodeY), modules: t.modules, span: t.span, margin: t.margin })),
    backs: {},
  };
}

// ai: The recorder's objects a tree names, and every object they name in turn (a pipeline's layouts), as the native
// ai: loader takes them: buffers with the hash of the bytes the host wrote (out/blobs/<hash>.bin, shared by every
// ai: buffer and variant holding the same bytes: the back half's tables are the same at every B), layouts, pipelines
// ai: by module hash.
function exportObjects(rec, tree, blobs) {
  const want = new Set();
  JSON.stringify(tree, (k, x) => { if (typeof x === "string" && /^[bLp]\d+$/.test(x)) want.add(x); return x; });
  for (const id of [...want]) { const o = rec.objects.get(id); if (o?.__kind === "p") for (const g of o.layout.groups) want.add(g.__id); }
  const buffers = {}, bgls = {}, pipelines = {};
  for (const id of [...want].sort((a, b) => a.localeCompare(b, "en", { numeric: true }))) {
    const o = rec.objects.get(id);
    if (!o) throw new Error(`the tree names ${id}, which the recorder never made`);
    if (o.__kind === "b") {
      const blob = o.data ? createHash("sha256").update(o.data).digest("hex").slice(0, 16) : null;
      if (blob && !blobs.has(blob)) blobs.set(blob, o.data);
      buffers[id] = { size: o.size, usage: o.usage, blob };
    } else if (o.__kind === "L") bgls[id] = o.entries;
    else if (o.__kind === "p") pipelines[id] = { module: o.module.hash, entry: o.entry, groups: o.layout.groups.map((g) => g.__id), label: o.label };
  }
  return { buffers, bgls, pipelines };
}

// ai: SPIR-V without the capabilities naga declares that a compute module never uses: StorageInputOutput16 (4436)
// ai: comes with every f16 module, and asks the device for 16-bit shader inputs and outputs, which a compute shader
// ai: has none of (its only inputs are u32 builtins) and some drivers lack. The NDK's spirv-opt (2022.4) has no
// ai: --trim-capabilities, so the instruction (OpCapability, opcode 17, two words) is cut here.
// ai: UniformAndStorageBuffer16BitAccess (4434) too: naga declares it with every f16 module, the S26 Ultra's Adreno 840
// ai: lacks the feature (its vkjson, 2026-09-29), and a compute module needs it only for a 16-bit type in a uniform
// ai: block. spirv-val does not check either (a module with f16 in a uniform validates without it: a negative control
// ai: run 2026-09-29), so the types are checked here: a capability is dropped only where no variable of its storage
// ai: class (Uniform for 4434, Input or Output for 4436) reaches a 16-bit type, and a module where one does throws.
const DROP_CAPS = new Map([[4436, [1, 3]], [4434, [2]]]);
export function stripCaps(path) {
  const b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4), out = Array.from(w.subarray(0, 5));
  const types = new Map(), vars = [];
  for (let i = 5; i < w.length;) {
    const n = w[i] >>> 16, op = w[i] & 0xffff;
    if (!n) throw new Error(`${path}: a zero-length instruction at word ${i}`);
    if (op === 21 || op === 22) types.set(w[i + 1], { small: w[i + 2] === 16, of: [] });                        // OpTypeInt, OpTypeFloat
    else if (op === 23 || op === 24 || op === 28 || op === 29) types.set(w[i + 1], { small: false, of: [w[i + 2]] });   // vector, matrix, array, runtime array
    else if (op === 30) types.set(w[i + 1], { small: false, of: Array.from(w.subarray(i + 2, i + n)) });          // OpTypeStruct
    else if (op === 32) types.set(w[i + 1], { small: false, of: [w[i + 3]], sc: w[i + 2] });                     // OpTypePointer
    else if (op === 59) vars.push({ type: w[i + 1], sc: w[i + 3] });                                             // OpVariable
    i += n;
  }
  const small = (id, seen = new Set()) => { if (seen.has(id)) return false; seen.add(id); const t = types.get(id); return !!t && (t.small || t.of.some((x) => small(x, seen))); };
  const needed = new Set();
  for (const [cap, classes] of DROP_CAPS) if (vars.some((v) => classes.includes(v.sc) && small(v.type))) needed.add(cap);
  if (needed.size) throw new Error(`${path}: a 16-bit type in a ${[...needed].map((c) => (c === 4434 ? "uniform block" : "shader input or output")).join(" and ")}: capability ${[...needed].join(", ")} needed, which the S26's Adreno lacks`);
  for (let i = 5; i < w.length;) {
    const n = w[i] >>> 16, op = w[i] & 0xffff;
    if (!(op === 17 && DROP_CAPS.has(w[i + 1]))) for (let k = 0; k < n; k++) out.push(w[i + k]);
    i += n;
  }
  writeFileSync(path, new Uint8Array(new Uint32Array(out).buffer));
}

// ai: The bitcast helpers nagaWgsl wrote (liz_h4, liz_h2, liz_p4, liz_p2: naga's WGSL has no bitcast between f16
// ai: vectors and u32s) given back their one instruction in the SPIR-V: each body becomes OpBitcast of its parameter,
// ai: which SPIR-V allows between types of one bit width (Tint emits the same). The unpack and pack they are
// ai: written with cost the f16 proposer 3.52 ms a frame on the iGPU against the web's 2.65 (2026-09-29). Found by
// ai: name (wgsl2spv writes names), their signature checked.
const BITCASTS = { liz_h4: ["v4f16", "v2u32"], liz_h2: ["v2f16", "u32"], liz_p4: ["v2u32", "v4f16"], liz_p2: ["u32", "v2f16"] };
export function patchBitcasts(path) {
  const b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  const str = (i, n) => { let s = ""; for (let k = i; k < i + n; k++) for (let j = 0; j < 4; j++) { const c = (w[k] >>> (8 * j)) & 255; if (!c) return s; s += String.fromCharCode(c); } return s; };
  const names = new Map(), types = new Map(), fnTypes = new Map();
  for (let i = 5; i < w.length;) {
    const n = w[i] >>> 16, op = w[i] & 0xffff;
    if (op === 5) names.set(w[i + 1], str(i + 2, n - 2));                                                      // OpName
    else if (op === 21) types.set(w[i + 1], `${w[i + 3] ? "i" : "u"}${w[i + 2]}`);                            // OpTypeInt
    else if (op === 22) types.set(w[i + 1], `f${w[i + 2]}`);                                                   // OpTypeFloat
    else if (op === 23) types.set(w[i + 1], `v${w[i + 3]}${types.get(w[i + 2])}`);                             // OpTypeVector
    else if (op === 33) fnTypes.set(w[i + 1], [w[i + 2], Array.from(w.subarray(i + 3, i + n))]);               // OpTypeFunction
    i += n;
  }
  let bound = w[3], patched = 0;
  const out = Array.from(w.subarray(0, 5));
  for (let i = 5; i < w.length;) {
    const n = w[i] >>> 16, op = w[i] & 0xffff;
    const want = op === 54 ? BITCASTS[names.get(w[i + 2])] : null;                                               // OpFunction
    if (!want) { for (let k = 0; k < n; k++) out.push(w[i + k]); i += n; continue; }
    const [ret, params] = fnTypes.get(w[i + 4]), name = names.get(w[i + 2]);
    if (types.get(ret) !== want[0] || params.length !== 1 || types.get(params[0]) !== want[1]) throw new Error(`${path}: ${name} is ${types.get(ret)}(${params.map((p) => types.get(p))}), not ${want[0]}(${want[1]})`);
    for (let k = 0; k < n; k++) out.push(w[i + k]);
    i += n;
    let param = -1, labeled = false;
    for (; (w[i] & 0xffff) !== 56; i += w[i] >>> 16) {                                                          // to OpFunctionEnd
      const op2 = w[i] & 0xffff;
      if (op2 === 55) { param = w[i + 2]; for (let k = 0; k < (w[i] >>> 16); k++) out.push(w[i + k]); }         // OpFunctionParameter
      else if (op2 === 248 && !labeled) {                                                                          // the first OpLabel
        labeled = true;
        out.push(w[i], w[i + 1]);
        const id = bound++;
        out.push((4 << 16) | 124, ret, id, param);                                                               // OpBitcast
        out.push((2 << 16) | 254, id);                                                                            // OpReturnValue
      }
    }
    out.push(w[i]);                                                                                                // OpFunctionEnd
    i += 1;
    patched++;
  }
  out[3] = bound;
  writeFileSync(path, new Uint8Array(new Uint32Array(out).buffer));
  return patched;
}

function run(cmd, args) {
  try { return execFileSync(cmd, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }); }
  catch (e) { const err = new Error(`${cmd.split("/").pop()} ${args.join(" ")}: ${(e.stderr || e.stdout || e.message).toString().trim().split("\n").slice(0, 12).join("\n")}`); err.tool = true; throw err; }
}

// ai: One module: the web's WGSL, naga's reading of it, then wgsl2spv to SPIR-V 1.3 (its options in its main.rs) with
// ai: naga's bound checks off (`--unchecked`: index, buffer and image, since 2026-09-30: an index out of range is
// ai: undefined behaviour here, where WebGPU clamps it), the capabilities no compute module uses cut, validated for
// ai: Vulkan 1.3 (workgroup memory zeroed by its initializer is 1.3's), optimised only with LIZ_SPIRV_OPT=1.
function compile(h, code) {
  const [w, n, raw, spv] = [`wgsl/${h}.wgsl`, `naga/${h}.wgsl`, `spv/${h}.raw.spv`, `spv/${h}.spv`].map((p) => fileURLToPath(new URL(p, OUT)));
  writeFileSync(w, code);
  writeFileSync(n, nagaWgsl(code));
  run(TOOLS.wgsl2spv, [n, raw, "--unchecked"]);
  stripCaps(raw);
  patchBitcasts(raw);
  run(TOOLS.val, ["--target-env", "vulkan1.3", raw]);
  if (OPT) { run(TOOLS.opt, ["-O", "--target-env=vulkan1.3", raw, "-o", spv]); run(TOOLS.val, ["--target-env", "vulkan1.3", spv]); }
  else writeFileSync(spv, readFileSync(raw));
  return { bytes: statSync(spv).size };
}

// ai: The gpu sources whose output this is: a change to any of them makes the setup stale (gen.json's `sources`
// ai: names what it was made from; nothing checks it at run time, so a change to gpu/ wants gen run again).
function sourcesHash() {
  const h = createHash("sha256");
  const walk = (dir) => { for (const f of readdirSync(dir).sort()) { const p = `${dir}/${f}`; if (statSync(p).isDirectory()) walk(p); else if (/\.(mjs|safetensors)$/.test(f)) h.update(f).update(readFileSync(p)); } };
  for (const d of ["gpu/wgsl", "gpu/back", "gpu/cnn", "gpu"]) { const dir = fileURLToPath(new URL(d, LIB)); for (const f of readdirSync(dir).sort()) { const p = `${dir}/${f}`; if (statSync(p).isFile() && /\.(mjs|safetensors)$/.test(f)) h.update(`${d}/${f}`).update(readFileSync(p)); } }
  walk(fileURLToPath(new URL("gpu/cnn/proposer", LIB))); walk(fileURLToPath(new URL("gpu/back/stop", LIB)));
  h.update(readFileSync(new URL("build/ob.wasm", LIB)));
  return h.digest("hex").slice(0, 16);
}

async function main() {
  for (const d of ["setup", "wgsl", "naga", "spv", "blobs"]) mkdirSync(new URL(`${d}/`, OUT), { recursive: true });
  const tables = await formatTables();
  const large = await loadNet(new URL(LARGE, LIB)), small = await loadNet(new URL(SMALL, LIB));
  const modules = new Map(), summary = [], blobs = new Map();
  for (const v of VARIANTS) {
    const rec = new Recorder(), log = [];
    const adapter = mockAdapter(rec, { features: FEATURES, limits: LIMITS });
    const fh = await FrontHalf.create(adapter, tables, {
      B: "auto", capMs: CAP_MS, inflight: "auto", restart: 0, cap: CAP, register: true, weights: large, precision: v.precision, subgroups: v.subgroups,
      bank: "fcn2", cascade: { weights: small, keep: CASCADE.keep, rest: CASCADE.rest }, back: "gpu", backN: 1536, fuse: true, log: (m) => log.push(m), probe: false, cancel: false, twins: null,
    });
    const tree = exportTree(fh, v, tables);
    tree.backs[fh.bh.B] = exportBack(fh.bh);
    for (const B of BACK_B.filter((b) => b !== fh.bh.B)) { await fh.buildBack(B, true); tree.backs[B] = exportBack(fh.bh); }
    tree.log = log.filter((m) => !m.startsWith("built in "));   // ai: no wall-clock line: two gens write the same tree
    const objects = exportObjects(rec, tree, blobs);
    tree.objects = objects;
    for (const p of Object.values(objects.pipelines)) if (!modules.has(p.module)) modules.set(p.module, rec.modules.get(p.module));
    writeFileSync(new URL(`setup/${v.name}.json`, OUT), JSON.stringify(tree));
    summary.push({ variant: v.name, pipelines: Object.keys(objects.pipelines).length, buffers: Object.keys(objects.buffers).length, nets: fh.nets });
    console.log(`${v.name}: ${Object.keys(objects.pipelines).length} pipelines, ${Object.keys(objects.buffers).length} buffers, nets ${JSON.stringify(fh.nets)}`);
  }
  let blobBytes = 0;
  for (const [h, data] of blobs) { writeFileSync(new URL(`blobs/${h}.bin`, OUT), data); blobBytes += data.byteLength; }
  console.log(`${blobs.size} blobs, ${(blobBytes / 1024).toFixed(0)} KB`);
  const compiled = {}, failed = [];
  for (const [h, code] of modules) {
    if (SKIP(code)) continue;
    try { compiled[h] = compile(h, code); } catch (e) { if (!e.tool) throw e; failed.push({ h, message: e.message }); }
  }
  // ai: The sender's GPU encoder (gen/sender.mjs, 2026-10-02): its kernels as out/spv/send_<name>.spv, its manifest
  // ai: setup/send.json and its bit map's table blobs/<hash>.bin, for core/tx/gpu_painter.cpp.
  const send = await senderSetup();
  writeFileSync(new URL(`blobs/${send.perm.hash}.bin`, OUT), send.perm.bytes);
  for (const [k, code] of Object.entries(send.modules)) {
    try { compiled[`send_${k}`] = compile(`send_${k}`, code); } catch (e) { if (!e.tool) throw e; failed.push({ h: `send_${k}`, message: e.message }); }
  }
  writeFileSync(new URL("setup/send.json", OUT), JSON.stringify(send.manifest));
  console.log(`sender: ${Object.keys(send.modules).length} kernels, PERMW ${(send.perm.bytes.byteLength / 1024).toFixed(0)} KB`);
  for (const f of failed) console.log(`FAIL ${f.h}\n${f.message}\n`);
  // ai: The native host's own shaders (core/ingest/*.comp, GLSL: the camera's ingest samples through a YCbCr
  // ai: sampler, which WGSL has no word for) to out/spv/ingest_<name>.spv by glslc.
  const ingestDir = new URL("../core/ingest/", HERE);
  for (const f of readdirSync(ingestDir).filter((f) => f.endsWith(".comp"))) {
    const out = fileURLToPath(new URL(`spv/ingest_${f.replace(/\.comp$/, "")}.spv`, OUT));
    run(TOOLS.glslc, ["--target-env=vulkan1.1", "-O", fileURLToPath(new URL(f, ingestDir)), "-o", out]);
    run(TOOLS.val, ["--target-env", "vulkan1.1", out]);
    console.log(`ingest ${f}: ${statSync(out).size} B of SPIR-V`);
  }
  // ai: The sender's own GLSL (core/tx/*.comp: the desktop presenter's expand of a ring frame into an RGBA8 storage
  // ai: image, which WGSL's bind kinds here do not cover) to out/spv/send_<name>.spv by glslc (2026-10-07).
  const txDir = new URL("../core/tx/", HERE);
  for (const f of readdirSync(txDir).filter((f) => f.endsWith(".comp"))) {
    const out = fileURLToPath(new URL(`spv/send_${f.replace(/\.comp$/, "")}.spv`, OUT));
    run(TOOLS.glslc, ["--target-env=vulkan1.1", "-O", fileURLToPath(new URL(f, txDir)), "-o", out]);
    run(TOOLS.val, ["--target-env", "vulkan1.1", out]);
    console.log(`sender ${f}: ${statSync(out).size} B of SPIR-V`);
  }
  const spvBytes = Object.values(compiled).reduce((a, c) => a + c.bytes, 0);
  console.log(`${modules.size} modules: ${Object.keys(compiled).length} compiled (${(spvBytes / 1024).toFixed(0)} KB of SPIR-V), ${failed.length} failed`);
  const naga = run(TOOLS.naga, ["--version"]).trim();
  writeFileSync(new URL("gen.json", OUT), JSON.stringify({ sources: sourcesHash(), naga, writer: "wgsl2spv (naga 30.0.1: zero-init native, no loop bounding, bound checks off)", spirvOpt: OPT, variants: summary, modules: Object.keys(compiled), failed: failed.map((f) => f.h) }, null, 1));
  if (failed.length) process.exit(1);
}

if (MAIN) await main();
