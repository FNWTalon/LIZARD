// ai: The native decoder's setup, taken from the decoder's setup host: liblizard/gpu/decoder.mjs FrontHalf.create,
// ai: with the options the native receiver runs (main's call below), is built on a recording device (mockgpu.mjs) once
// ai: a variant, and what it made is written out: every shader module's GLSL (gpu/glsl/) compiled to SPIR-V by glslang
// ai: (`compileGlsl`), every bind group layout and pipeline, every static buffer with the bytes the host wrote into it,
// ai: and the values the lanes and batches are made from (`exportTree`). The native host (core/dec/) builds exactly
// ai: these objects and is the one that runs lanes and batches. Nothing here decodes; the tables come from the codec's
// ai: wasm module (sim/ob.mjs over liblizard/build/ob.mjs).
//
// ai: The sender's GPU painter (gen/sender.mjs) is the browser sender's WGSL (gpu/wgsl/send.mjs, inverse.mjs), read by
// ai: naga (rewritten where naga reads WGSL otherwise than Tint, `nagaWgsl`) into the same out/spv. The decoder's WGSL
// ai: twins went 2026-10-31 (archive/webgpu-decoder/).
//
// ai: A variant is (precision, subgroups): precision int8 (the int8 proposer, the classifier f16 with its int8 twin
// ai: built beside), f16 or f32, subgroups whether the classifier reduces by shuffles. The back half is built at every
// ai: batch size a device may need (B_CEIL down to 1: the host builds it at Bmax where memory holds fewer frames), all
// ai: at workgroup memory 32 KB, the floor every phone we know has. The anchors' stages (gpu/glsl/anchors.mjs, SPEC
// ai: 6.13) are a setup's `anch` (exportAnch), built with the front half; pass 1, every back half's, reads their
// ai: records.
//
//   node liblizard/gen/gen.mjs [out]      (lizard-android/build.sh gen; out defaults to liblizard/out)
// ai: Writes out/setup/<variant>.json, out/blobs/<hash>.bin (the buffers' bytes), out/glsl/<hash>.comp (a decoder
// ai: module), out/wgsl/<hash>.wgsl and out/naga/<hash>.wgsl (a sender's module as written and as naga reads it) and
// ai: out/spv/<hash>.spv, out/webgpu/<hash>.wgsl (the plain variant's kernels as WGSL for WebGPU, webgpuOf), and
// ai: out/gen.json (the variants, the modules, the GLSL ones among them, the WebGPU ones, the gpu sources' hash that made
// ai: them). Needs, each checked before anything runs: ../build/ob.mjs and the module it names,
// ai: ../bindings/wasm/dist/lizard.wasm (liblizard's ./build.sh); wgsl2spv and glslang 16.6.0 (lizard-android/build.sh
// ai: tools; glslang at liblizard/.tools/glslang, else the PATH's); SPIRV-Tools' spirv-val and shaderc's glslc, the
// ai: Android NDK's or else the PATH's.
import { installGlobals, Recorder, mockAdapter } from "./mockgpu.mjs";
import { senderSetup } from "./sender.mjs";
import { mkdirSync, writeFileSync, readFileSync, existsSync, readdirSync, statSync, realpathSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { zstdCompressSync, constants as zlibConstants } from "node:zlib";
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

// ai: The tree's version, every setup's treeVersion, which the native host (core/dec/front.cpp TREE_VERSION) refuses a
// ai: tree of another of: raised in both places whenever what the host reads of a tree changes.
// ai: 5 since 2026-10-31: the decoder's WGSL archived and F9's three dead buffers dropped (every later buffer id moved),
// ai: pass 1's bilinear reads by the texture unit (a layout's `sampler` entry: the device's immutable linear sampler),
// ai: the soft and LDPC steps dispatched from the gate's ARGS (each rule step's `argsAt`); 4 the LDPC pipelines'
// ai: `lanes`; 3 REDO's R2 args; 2 PEAKS; 1 from 2026-10-29. 6 since 2026-10-31: a pipeline's `needs` (the device
// ai: features its module asks beyond the variant's: "int16", the LDPC's "reg+i16" forms). 7 since 2026-11-02: the
// ai: front's `trace` (TRACE, each quad's bend, in both rounds' grids) and HYP doubled (SCORE's bend bit).
// ai: 8 since 2026-11-06: REFIT's and REFIT_POST's layouts seven bindings (the eighth, REFIT's debug words, gone).
export const TREE_VERSION = 8;

const NDK = process.env.ANDROID_NDK_HOME || `${process.env.ANDROID_HOME || `${homedir()}/Android/Sdk`}/ndk/27.1.12297006`;
// ai: SPIRV-Tools' spirv-val and shaderc's glslc: the Android NDK's shader-tools where there is an NDK, else the
// ai: system's on the PATH
const EXE = process.platform === "win32" ? ".exe" : "";
const shaderTool = (name) => {
  const ndk = join(NDK, "shader-tools", { darwin: "darwin-x86_64", win32: "windows-x86_64" }[process.platform] ?? "linux-x86_64", name + EXE);
  if (existsSync(ndk)) return ndk;
  for (const dir of (process.env.PATH ?? "").split(delimiter)) if (dir && existsSync(join(dir, name + EXE))) return join(dir, name + EXE);
  return ndk;
};
// ai: wgsl2spv (gen/wgsl2spv, naga 30 as a library: lizard-android/build.sh tools builds it) in place of naga-cli, whose
// ai: writer zeroes workgroup memory from one invocation and bounds every loop (wgsl2spv/src/main.rs says what that
// ai: cost). The SPIR-V reaches the driver as it validates: spirv-opt -O unrolled the fused pass 1 and F8's word to three
// ai: times their size (2026-09-29).
// ai: glslang for the decoder's GLSL modules: liblizard/.tools/glslang (lizard-android/build.sh tools puts the release
// ai: build there) where it is, else the PATH's. Not the NDK's glslc (shaderc 2022.3 on an older glslang), which
// ai: compiles only the host's own ingest and expand. Pinned: a module is named by its source, so another glslang would
// ai: write other SPIR-V under the same names; a run refuses any but GLSLANG (build.sh tools's GLSLANG too).
export const GLSLANG = "16.6.0";
const glslangVersion = (p) => { try { return execFileSync(p, ["--version"], { encoding: "utf8" }).split("\n")[0].split(":").pop().trim(); } catch { return null; } };
const glslangTool = () => {
  const lab = fileURLToPath(new URL(`../.tools/glslang/bin/glslang${EXE}`, HERE));
  if (existsSync(lab)) return lab;
  for (const dir of (process.env.PATH ?? "").split(delimiter)) if (dir && existsSync(join(dir, "glslang" + EXE))) return join(dir, "glslang" + EXE);
  return lab;
};
export const TOOLS = { wgsl2spv: fileURLToPath(new URL(`./wgsl2spv/target/release/wgsl2spv${EXE}`, HERE)), spv2wgsl: fileURLToPath(new URL(`./wgsl2spv/target/release/spv2wgsl${EXE}`, HERE)), glslang: glslangTool(), val: shaderTool("spirv-val"), glslc: shaderTool("glslc") };

// ai: What a run needs, checked before the gpu modules load (sim/ob.mjs imports ../build/ob.mjs, which names the
// ai: codec's module): every one missing, each with where it comes from.
const WASM = new URL("bindings/wasm/dist/lizard.wasm", LIB);
if (MAIN) {
  const ndk = (or) => `the Android NDK's shader-tools: ANDROID_NDK_HOME; or ${or} on the PATH`;
  const missing = [
    [fileURLToPath(new URL("build/ob.mjs", LIB)), "liblizard/build.sh"], [fileURLToPath(WASM), "liblizard/build.sh"],
    [TOOLS.wgsl2spv, "lizard-android/build.sh tools"], [TOOLS.glslang, `glslang ${GLSLANG}: lizard-android/build.sh tools, or glslang on the PATH`],
    [TOOLS.val, ndk("SPIRV-Tools")], [TOOLS.glslc, ndk("shaderc")],
  ].filter(([p]) => !existsSync(p));
  if (missing.length) { console.error(missing.map(([p, from]) => `no ${p} (${from})`).join("\n")); process.exit(1); }
  const v = glslangVersion(TOOLS.glslang);
  if (v !== GLSLANG) { console.error(`${TOOLS.glslang} is glslang ${v ?? "of no version it states"}; the tree's modules are glslang ${GLSLANG}'s (lizard-android/build.sh tools)`); process.exit(1); }
}

const { FrontHalf, CAP_MS, B_CEIL, LEVELS, COUNTERS, TAU, EPS, INSTALLED, receiverOptions, RESULT, SEL_WORDS, TS_MAX, TS_PLAIN, RING_SLACK, TWIN_FRAMES, TWIN_ROUNDS } =
  await imp("gpu/decoder.mjs");
const { formatTables, PICTURES } = await imp("gpu/tables.mjs");
const { loadNet } = await imp("gpu/cnn/netfile.mjs");
const { K, CELL, HYP, RING_COUNT, VOTE_READS, PEAK_LIST, R2_N } = await imp("gpu/plan/finder.mjs");
const { NODES_MAX, RESID } = await imp("gpu/plan/register.mjs");
const { ANCH_K, ANCH_REC } = await imp("gpu/plan/anchors.mjs");
const { PYRAMID_BLOCK } = await imp("gpu/plan/pyramid.mjs");
const { TILE } = await imp("gpu/plan/bank_fcn2.mjs");
const { SLOTS } = await imp("gpu/plan/common.mjs");
const { ARGS_WORDS, ARGS_SLOT, argsWords } = await imp("gpu/plan/back_transform.mjs");
const { VERSION_MIN, VERSION_MAX } = await imp("gpu/wordcode.mjs");
const { COUNTS } = await imp("gpu/back/back.mjs");
const { REC_BYTES } = await imp("gpu/back/ldpc.mjs");
const { PAYLOAD } = await imp("gpu/back/ref_ldpc.mjs");

// ai: The classifier the native receiver runs (gpu/decoder.mjs INSTALLED): cnn-v3 on every peak, a path under
// ai: liblizard/ (as the app's build needs it).
const NET = INSTALLED.classifier;
// ai: The device the setup is made for: 32 KB of workgroup memory (every phone we know, and what Chrome gives the
// ai: desktop's GPUs), the rest generous; plan()'s memory limits are the real device's, read at run time.
const LIMITS = { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31, maxComputeWorkgroupStorageSize: 32768, maxStorageBuffersPerShaderStage: 16, maxTextureArrayLayers: 2048, maxComputeInvocationsPerWorkgroup: 256 };
const FEATURES = ["timestamp-query", "shader-f16", "subgroups"];
export const VARIANTS = ["int8", "f16", "f32"].flatMap((precision) => [true, false].map((subgroups) => ({ precision, subgroups, name: `${precision}${subgroups ? "-sg" : ""}` })));
// ai: the variant whose kernels WebGPU's core takes (no 8- or 16-bit types, no subgroups), out/webgpu's (webgpuOf)
const WEBGPU_VARIANT = "f32";
const BACK_B = [B_CEIL, 16, 8, 4, 2, 1];

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
      built: t.built, cbytes: t.cbytes, yStride: t.yStride, sStride: t.sStride, partStride: t.partStride,
      gate: idOf(t.gate), gateBgl: idOf(t.gateBgl), p1fBgl: idOf(t.p1fBgl), redBgl: idOf(t.redBgl), p2Bgl: idOf(t.p2Bgl),
      pass1f: perSlot(t.pass1f), reduce: perSlot(t.reduce), pass2: perSlot(t.pass2),
      twBufs: perSlot(t.twBufs), x12Bufs: perSlot(t.x12Bufs), rowsBufs: perSlot(t.rowsBufs), picUnis: perSlot(t.picUnis), gateUni: idOf(t.gateUni),
      sizes: t.tables.map((tb) => (tb ? { n: tb.n, subch: tb.subch, blocks: tb.blocks, m2: tb.m2, sx2: tb.sx2, sq2: tb.sq2 } : null)),
    },
    // ai: the soft and LDPC stages' tables and lane buffers (back/soft.mjs, back/ldpc.mjs), which the rate profile's
    // ai: stage below decodes through
    soft: { uvBuf: idOf(so.uvBuf), params: perSlot(so.params), served: so.served, blocks: so.sizes.map((z) => (z ? z.blocks : 0)), bytes: so.bytes },
    ldpc: { bytes: ld.bytes, blocksMax: ld.blocksMax, recCap: ld.recCap },
    // ai: the format's rate profile's stage (gpu/back/tiers.mjs exportFor), in the soft and LDPC stages' place
    rule: bh.rule.exportFor(idOf),
  };
}

// ai: What the native host's lanes and batches read of the front half (core/dec/front.cpp), with the host's
// ai: constants, so the native code restates none of them.
function exportTree(fh, v) {
  const tf = fh.twinForms;
  const twin = (t) => (t ? { pipe: pbNet(t.pipe), buf: idOf(t.buf), precision: t.w.precision } : null);
  const f = fh.fcn2Form;
  return {
    treeVersion: TREE_VERSION, variant: v.name, precision: fh.precision, floatPrecision: fh.floatPrecision, subgroups: v.subgroups, features: fh.features, nets: fh.nets,
    consts: { LEVELS, COUNTERS, RESULT, SEL_WORDS, TS_MAX, TS_PLAIN, RING_SLACK, K, CELL, HYP, RING_COUNT, VOTE_READS, PEAK_LIST, R2_N, NODES_MAX, RESID, PYRAMID_BLOCK, TAU, EPS,
      TILE, B_CEIL, CAP_MS, TWIN_FRAMES, TWIN_ROUNDS, VERSION_MIN, VERSION_MAX, SLOTS, ARGS_WORDS, ARGS_SLOT, argsWords, COUNTS, REC_BYTES, PAYLOAD, ANCH_K, ANCH_REC },
    cap: fh.cap, nodesMax: fh.nodesMax, modules: Array.from(fh.modules), bankKeep: fh.bank.KEEP,
    weightsPrecision: fh.weights.precision,
    front: {
      pyramid: pb(fh.pyramid), classify: pbNet(fh.classify),
      vote: pb(fh.vote), peaks: pb(fh.peaks), gather: pb(fh.gather), complete: pb(fh.complete), score: pb(fh.score), pick: pb(fh.pick), trace: pb(fh.trace), nodesP: pb(fh.nodesP), refit: pb(fh.refit), refitPost: pb(fh.refitPost), wordP: pb(fh.wordP), select: pb(fh.select),
      // ai: the finder's second round (gpu/glsl/finder.mjs REDO; gpu/bank_fcn2.mjs COARSE_FROM)
      redo: pb(fh.redo), coarseFrom: fh.bank.COARSE_FROM,
      fcn2: pb(fh.fcn2), fcn2X: pb(fh.fcn2X), fcn2Form: { TX: f.TX, TY: f.TY, XB: f.XB, walk: f.walk, xplane: f.xplane, int8: f.int8, store: f.store, name: f.name },
      buffers: Object.fromEntries(["planBuf", "planUni", "p4Uni", "pickUni", "nodesBuf", "tmplBuf", "wordPtsBuf", "wordUni", "wordLatUni", "fitUni", "weightsBuf"].map((k) => [k, idOf(fh[k])])),
      round: idOf(fh.round),
      twins: tf ? { large: twin(tf.large) } : null,
    },
    pictures: PICTURES.map(({ n, subch, lo, hi }) => ({ n, subch, lo, hi })),
    backs: {},
  };
}

// ai: The anchors' stages (gpu/glsl/anchors.mjs), which the native host builds with the front half (core/dec/setup.cpp
// ai: build): ANCH_FIND and ANCH_FIT ({ p, bgl }) and their uniform (anchUni).
function exportAnch(fh) {
  return {
    // ai: refit: ANCH_REFIT ({ p, bgl }), which runs between ANCH_FIND and ANCH_FIT; predict: ANCH_PREDICT on its layout,
    // ai: and nodes: the border's second search (F7's NODES centred) on NODES' layout with its own round
    // ai: (buffers.round), both run before it
    find: pb(fh.anchFind), predict: pb(fh.anchPredict), nodes: pb(fh.nodesC), refit: pb(fh.anchRefit), fit: pb(fh.anchFit),
    buffers: { anchUni: idOf(fh.anchUni), round: idOf(fh.anchRound) },
  };
}

// ai: The recorder's objects a tree names, and every object they name in turn (a pipeline's layouts), as the native
// ai: loader takes them: buffers with the hash of the bytes the host wrote (out/blobs/<hash>.bin, shared by every
// ai: buffer and variant holding the same bytes: the back half's tables are the same at every B), layouts, pipelines
// ai: by module hash.
// ai: A pipeline's lanes, its workgroup's invocations (the product of the module's @workgroup_size, or a GLSL module's
// ai: local_size_x, _y and _z, where they are literal): the host leaves out a pipeline past its device's
// ai: maxComputeWorkGroupInvocations (core/dec/setup.cpp).
function lanesOf(code) {
  const m = /@workgroup_size\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?(?:,\s*(\d+)\s*)?\)/.exec(code ?? "");
  if (m) return { lanes: Number(m[1]) * Number(m[2] ?? 1) * Number(m[3] ?? 1) };
  const g = /layout\s*\(([^)]*local_size_x[^)]*)\)\s*in\s*;/.exec(code ?? "");
  if (!g) return {};
  const at = (k) => { const v = new RegExp(`local_size_${k}\\s*=\\s*(\\d+)`).exec(g[1]); return v ? Number(v[1]) : 1; };
  return { lanes: at("x") * at("y") * at("z") };
}
// ai: The device features a module needs beyond its variant's, by the extension that brings each into its GLSL
// ai: (gpu/glsl/common.mjs): "int16" (GL_EXT_shader_explicit_arithmetic_types_int16, SPIR-V Int16: shaderInt16). The
// ai: host leaves out a pipeline whose device lacks one (core/dec/setup.cpp); compileGlsl holds the SPIR-V to it.
const NEEDS = [["int16", /GL_EXT_shader_explicit_arithmetic_types_int16\b/, 22]];   // ai: the capability's number, Int16
function needsOf(code) {
  const needs = NEEDS.filter(([, re]) => re.test(code ?? "")).map(([k]) => k);
  return needs.length ? { needs } : {};
}
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
    else if (o.__kind === "p") pipelines[id] = { module: o.module.hash, entry: o.entry, groups: o.layout.groups.map((g) => g.__id), label: o.label, ...lanesOf(o.module.code), ...needsOf(o.module.code) };
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

// ai: The workgroup size declared as SPIR-V's WorkgroupSize builtin constant too, equal to the entry point's LocalSize, as
// ai: glslc writes it (2026-10-23, the sender on macOS): Metal has no initializer for threadgroup memory, so MoltenVK's
// ai: SPIRV-Cross zeroes a workgroup variable with a loop strided by gl_WorkGroupSize, which it declares only from that
// ai: constant, and the kernel would not compile ("use of undeclared identifier 'gl_WorkGroupSize'"). Only where a
// ai: workgroup variable has an initializer (naga's Native zeroing) and no such constant exists; 0 where nothing changed.
export function declareWorkgroupSize(path) {
  const b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  let size = null, uint = 0, uvec3 = 0, zeroed = false, firstType = -1, firstFn = -1;
  for (let i = 5; i < w.length;) {
    const n = w[i] >>> 16, op = w[i] & 0xffff;
    if (op === 16 && w[i + 2] === 17) size = [w[i + 3], w[i + 4], w[i + 5]];                                    // OpExecutionMode LocalSize
    else if (op === 71 && w[i + 2] === 11 && w[i + 3] === 25) return 0;                                         // BuiltIn WorkgroupSize already
    else if (op === 21 && w[i + 2] === 32 && w[i + 3] === 0) uint = w[i + 1];                                   // OpTypeInt 32 0
    else if (op === 59 && w[i + 3] === 4 && n === 5) zeroed = true;                                             // a Workgroup OpVariable with initializer
    if (firstType < 0 && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op === 59)) firstType = i;
    if (firstFn < 0 && op === 54) firstFn = i;                                                                  // OpFunction
    i += n;
  }
  for (let i = 5; i < w.length && uint; i += w[i] >>> 16) if ((w[i] & 0xffff) === 23 && w[i + 2] === uint && w[i + 3] === 3) uvec3 = w[i + 1];
  if (!zeroed || !size || firstType < 0 || firstFn < 0) return 0;
  let bound = w[3];
  const decl = [], ids = { uint: uint || bound++ };
  if (!uint) decl.push((3 << 16) | 21, ids.uint, 32, 0);                                                        // OpTypeInt 32 0
  ids.uvec3 = uvec3 || bound++;
  if (!uvec3) decl.push((4 << 16) | 23, ids.uvec3, ids.uint, 3);                                                // OpTypeVector
  const c = size.map((v) => { const id = bound++; decl.push((4 << 16) | 43, ids.uint, id, v); return id; });  // OpConstant
  const wgs = bound++;
  decl.push((6 << 16) | 44, ids.uvec3, wgs, ...c);                                                              // OpConstantComposite
  const out = [...w.subarray(0, firstType), (4 << 16) | 71, wgs, 11, 25, ...w.subarray(firstType, firstFn), ...decl, ...w.subarray(firstFn)];
  out[3] = bound;
  writeFileSync(path, new Uint8Array(new Uint32Array(out).buffer));
  return 1;
}

function run(cmd, args) {
  try { return execFileSync(cmd, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }); }
  catch (e) { const err = new Error(`${cmd.split("/").pop()} ${args.join(" ")}: ${(e.stderr || e.stdout || e.message).toString().trim().split("\n").slice(0, 12).join("\n")}`); err.tool = true; throw err; }
}

// ai: A sender's module (gen/sender.mjs): its WGSL, naga's reading of it, then wgsl2spv to SPIR-V 1.3 (its options in its main.rs) with
// ai: naga's bound checks off (`--unchecked`: index, buffer and image, since 2026-09-30: an index out of range is
// ai: undefined behaviour here, where WebGPU clamps it), the capabilities no compute module uses cut, validated for
// ai: Vulkan 1.3 (workgroup memory zeroed by its initializer is 1.3's), its workgroup size declared as the builtin
// ai: constant (declareWorkgroupSize, for MoltenVK). A decoder module (GLSL): compileGlsl.
function compile(h, { lang, code }) {
  if (lang === "glsl") return compileGlsl(h, code);
  const [w, n, raw, spv] = [`wgsl/${h}.wgsl`, `naga/${h}.wgsl`, `spv/${h}.raw.spv`, `spv/${h}.spv`].map((p) => fileURLToPath(new URL(p, OUT)));
  writeFileSync(w, code);
  writeFileSync(n, nagaWgsl(code));
  run(TOOLS.wgsl2spv, [n, raw, "--unchecked"]);
  stripCaps(raw);
  patchBitcasts(raw);
  if (h.startsWith("send_")) declareWorkgroupSize(raw);
  run(TOOLS.val, ["--target-env", "vulkan1.3", raw]);
  writeFileSync(spv, readFileSync(raw));
  return { bytes: statSync(spv).size };
}

// ai: A GLSL module (gpu/glsl/): glslang for Vulkan 1.1, which is SPIR-V 1.3, naga's version, so a device of the
// ai: host's Vulkan 1.1 floor loads either; no optimiser (the SPIR-V reaches the driver as written, as naga's does);
// ai: bound checks are GLSL's own, none. Its capabilities checked as naga's are (stripCaps: a 16-bit type in a
// ai: uniform block or an input refused), validated for Vulkan 1.3 as naga's (a workgroup variable a GLSL source
// ai: zeroes, GL_EXT_null_initializer, is 1.3's or its extension's, as naga's zeroing is).
function compileGlsl(h, code) {
  const [g, raw, spv] = [`glsl/${h}.comp`, `spv/${h}.raw.spv`, `spv/${h}.spv`].map((p) => fileURLToPath(new URL(p, OUT)));
  writeFileSync(g, code);
  run(TOOLS.glslang, ["-V", "--target-env", "vulkan1.1", "-S", "comp", "--quiet", "-o", raw, g]);
  stripCaps(raw);
  // ai: a capability a pipeline's `needs` names is declared exactly where its source asks for it
  const caps = capsOf(raw);
  for (const [k, re, cap] of NEEDS) if (caps.has(cap) !== re.test(code)) throw new Error(`${g}: capability ${cap} ${caps.has(cap) ? "declared without" : "not declared with"} its extension: the pipeline's needs ("${k}") would be wrong`);
  run(TOOLS.val, ["--target-env", "vulkan1.3", raw]);
  writeFileSync(spv, readFileSync(raw));
  return { bytes: statSync(spv).size, glsl: true };
}

// ai: The plain variant's kernels as WGSL for WebGPU (2026-11-06; the decoder's WGSL twins went 2026-10-31, the GLSL the
// ai: one source): a GLSL module compiled again for WGSL's sake, then naga's SPIR-V front end and its WGSL writer
// ai: (wgsl2spv's spv2wgsl). Two things naga 30's reader does not take, met here: a workgroup variable zeroed by its
// ai: initializer and then used atomically (the `= {}` dropped: WGSL zeroes workgroup memory itself), and an arrayed
// ai: image's integer coordinates where the module declares no 2-vector of that integer (ensureCoordTypes). A module it
// ai: cannot write (a form on 16-bit integers, which WGSL lacks) is listed in gen.json, not a failure of the run.
function webgpuOf(h, code) {
  const [g, spv, w] = [`webgpu/${h}.comp`, `webgpu/${h}.spv`, `webgpu/${h}.wgsl`].map((p) => fileURLToPath(new URL(p, OUT)));
  writeFileSync(g, code.replace(/^(\s*shared\s+[^;=]+?)\s*=\s*\{\s*\}\s*;/gm, "$1;"));
  try {
    run(TOOLS.glslang, ["-V", "--target-env", "vulkan1.1", "-S", "comp", "--quiet", "-o", spv, g]);
    ensureCoordTypes(spv);
    run(TOOLS.spv2wgsl, [spv, w]);
    writeFileSync(w, uniformBuiltins(readFileSync(w, "utf8")));
    return { bytes: statSync(w).size };
  } finally { rmSync(g, { force: true }); rmSync(spv, { force: true }); }
}

// ai: WGSL proves a barrier's control flow uniform (Tint refuses a module it cannot: 100 of the 141 here, 2026-11-06),
// ai: and naga's SPIR-V reader hands the entry's builtins to the body (main_1) through private variables, whose every
// ai: read Tint counts non-uniform: a workgroup's early return on its own workgroup_id put every barrier after it out.
// ai: The builtins uniform over a workgroup (workgroup_id, num_workgroups) are given to main_1 as parameters named as
// ai: the private variables, which they shadow there, so the body reads them as written and the analysis follows them
// ai: in from the entry; any other function still reads the private copy, which the entry still writes.
const UNIFORM_BUILTINS = new Set(["workgroup_id", "num_workgroups"]);
// ai: The GLSL's way to hand a workgroup one invocation's word (`barrier(); uint d = flag; barrier();`), which naga writes as
// ai: a barrier and then a read of the workgroup variable: Tint counts that read non-uniform, so a branch on it puts every
// ai: barrier under it out. A read of a workgroup variable of a plain type straight after a barrier (every invocation is
// ai: there) is made WGSL's workgroupUniformLoad, a barrier and a read Tint counts uniform.
function uniformReads(code) {
  const plain = "(?:u32|i32|f32|bool|vec[234]<(?:u32|i32|f32)>)";
  const shared = new Set([...code.matchAll(new RegExp(`^var<workgroup> (\\w+): (?:${plain}|array<${plain}, \\d+>);$`, "gm"))].map((m) => m[1]));
  if (!shared.size) return code;
  const lines = code.split("\n");
  for (let i = 1; i < lines.length; i++) {
    // ai: the variable, or one element of it at a constant index
    const m = lines[i].match(/^(\s*)let (_e\d+) = (\w+)(\[\d+[iu]?\])?;$/);
    if (m && shared.has(m[3]) && lines[i - 1].trim() === "workgroupBarrier();") lines[i] = `${m[1]}let ${m[2]} = workgroupUniformLoad(&${m[3]}${m[4] ?? ""});`;
  }
  return lines.join("\n");
}
function uniformBuiltins(code) {
  const entry = code.match(/fn main\(([^\n]*)\) \{\n((?:[ \t]+\w+ = \w+;\n)*)([ \t]+)main_1\(\);\n\}/);
  code = uniformReads(code);
  if (!entry || !/\nfn main_1\(\) \{/.test(code)) return code;
  const params = [...entry[1].matchAll(/@builtin\((\w+)\) (\w+): ([\w<>]+)/g)].filter((m) => UNIFORM_BUILTINS.has(m[1]));
  const given = params.map(([, , name, type]) => ({ name, type, global: entry[2].match(new RegExp(`(\\w+) = ${name};`))?.[1] })).filter((p) => p.global);
  if (!given.length) return code;
  const wrapper = entry[0].replace(/main_1\(\);/, `main_1(${given.map((p) => p.name).join(", ")});`);
  return code.replace(entry[0], wrapper).replace(/\nfn main_1\(\) \{/, `\nfn main_1(${given.map((p) => `${p.global}: ${p.type}`).join(", ")}) {`);
}

// ai: naga's SPIR-V reader takes an arrayed image's integer coordinates apart by a 2-vector of that integer it finds
// ai: among the module's types (front/spv/image.rs, extract_image_coordinates), which glslang declares only where the
// ai: source uses one: each 32-bit integer's 2-vector declared where it is missing (an unused type, the id bound raised).
function ensureCoordTypes(path) {
  const b = readFileSync(path), w = Array.from(new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4));
  const ints = new Map(), vec2 = new Set();
  for (let i = 5; i < w.length; i += w[i] >>> 16 || 1) {
    const op = w[i] & 0xffff;
    if (op === 21 && w[i + 2] === 32) ints.set(w[i + 1], i);   // ai: OpTypeInt 32 (either signedness)
    if (op === 23 && w[i + 3] === 2) vec2.add(w[i + 2]);         // ai: OpTypeVector of 2
  }
  const add = [...ints].filter(([id]) => !vec2.has(id)).sort((a, b) => b[1] - a[1]);
  if (!add.length) return;
  for (const [id, at] of add) w.splice(at + (w[at] >>> 16), 0, (4 << 16) | 23, w[3]++, id, 2);
  writeFileSync(path, new Uint8Array(new Uint32Array(w).buffer));
}

// ai: A SPIR-V module's capabilities (OpCapability, opcode 17).
function capsOf(path) {
  const b = readFileSync(path), w = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4), caps = new Set();
  for (let i = 5; i < w.length; i += w[i] >>> 16 || 1) if ((w[i] & 0xffff) === 17) caps.add(w[i + 1]);
  return caps;
}

// ai: The gpu sources whose output this is: a change to any of them makes the setup stale (gen.json's `sources`
// ai: names what it was made from; nothing checks it at run time, so a change to gpu/ wants gen run again).
function sourcesHash() {
  const h = createHash("sha256");
  const walk = (dir) => { for (const f of readdirSync(dir).sort()) { const p = `${dir}/${f}`; if (statSync(p).isDirectory()) walk(p); else if (/\.(mjs|safetensors)$/.test(f)) h.update(f).update(readFileSync(p)); } };
  for (const d of ["gpu/plan", "gpu/wgsl", "gpu/glsl", "gpu/back", "gpu/cnn", "gpu"]) { const dir = fileURLToPath(new URL(d, LIB)); for (const f of readdirSync(dir).sort()) { const p = `${dir}/${f}`; if (statSync(p).isFile() && /\.(mjs|safetensors)$/.test(f)) h.update(`${d}/${f}`).update(readFileSync(p)); } }
  walk(fileURLToPath(new URL("gpu/cnn/proposer", LIB))); walk(fileURLToPath(new URL("gpu/back/stop", LIB)));
  h.update(readFileSync(WASM));
  return h.digest("hex").slice(0, 16);
}

// ai: The WebGPU tree as the web loads it (gpu/host.mjs over gpu/webgpu.zst, 2026-11-06): WEBGPU_VARIANT's setup, the
// ai: blobs it names and its pipelines' WGSL, each file under the path the host reads it by, in one container ("LZWT",
// ai: u32 1, u32 count, then a file each: u16 name length, the name in UTF-8, u32 size, the bytes; little-endian),
// ai: as one zstd frame at level 19 with long matching (its content size in the frame, as the codec's zstd asks). The
// ai: web takes it by a literal name, so the installed app hashes and caches it as any asset, and it is in git: the
// ai: site's build has no glslang or naga to make it. A pipeline whose module has no WGSL (the int16 forms) has none here.
export const WEBGPU_TREE = new URL("../gpu/webgpu.zst", HERE);
function packWebgpu(wgslModules) {
  const has = new Set(wgslModules);
  const setup = `setup/${WEBGPU_VARIANT}.json`;
  const tree = JSON.parse(readFileSync(new URL(setup, OUT), "utf8"));
  const files = [setup];
  for (const b of new Set(Object.values(tree.objects.buffers).map((b) => b.blob).filter(Boolean))) files.push(`blobs/${b}.bin`);
  for (const m of new Set(Object.values(tree.objects.pipelines).map((p) => p.module))) if (has.has(m)) files.push(`webgpu/${m}.wgsl`);
  const parts = [];
  const head = Buffer.alloc(12);
  head.write("LZWT", 0, "latin1"); head.writeUInt32LE(1, 4); head.writeUInt32LE(files.length, 8);
  parts.push(head);
  for (const f of files) {
    const name = Buffer.from(f, "utf8"), data = readFileSync(new URL(f, OUT)), e = Buffer.alloc(6 + name.length);
    e.writeUInt16LE(name.length, 0); name.copy(e, 2); e.writeUInt32LE(data.length, 2 + name.length);
    parts.push(e, data);
  }
  const raw = Buffer.concat(parts);
  const z = zstdCompressSync(raw, { params: { [zlibConstants.ZSTD_c_compressionLevel]: 19, [zlibConstants.ZSTD_c_enableLongDistanceMatching]: 1, [zlibConstants.ZSTD_c_windowLog]: 27 }, pledgedSrcSize: raw.length });
  writeFileSync(WEBGPU_TREE, z);
  console.log(`webgpu tree: ${files.length} files, ${(raw.length / 1e6).toFixed(2)} MB, ${(z.length / 1e6).toFixed(2)} MB packed (${fileURLToPath(WEBGPU_TREE)})`);
}

async function main() {
  for (const d of ["setup", "wgsl", "naga", "glsl", "spv", "blobs", "webgpu"]) mkdirSync(new URL(`${d}/`, OUT), { recursive: true });
  const tables = await formatTables();
  const net = await loadNet(new URL(NET, LIB));
  const modules = new Map(), summary = [], blobs = new Map(), used = {};
  for (const v of VARIANTS) {
    const rec = new Recorder(), log = [];
    const adapter = mockAdapter(rec, { features: FEATURES, limits: LIMITS });
    const fh = await FrontHalf.create(adapter, tables, {
      ...receiverOptions(net), precision: v.precision, subgroups: v.subgroups, backN: 1536, log: (m) => log.push(m), twins: null,
    });
    const tree = exportTree(fh, v);
    tree.backs[fh.bh.B] = exportBack(fh.bh);
    for (const B of BACK_B.filter((b) => b !== fh.bh.B)) { await fh.buildBack(B, true); tree.backs[B] = exportBack(fh.bh); }
    tree.anch = exportAnch(fh);
    tree.log = log;
    const objects = exportObjects(rec, tree, blobs);
    tree.objects = objects;
    for (const p of Object.values(objects.pipelines)) if (!modules.has(p.module)) modules.set(p.module, rec.modules.get(p.module));
    used[v.name] = new Set(Object.values(objects.pipelines).map((p) => p.module));
    writeFileSync(new URL(`setup/${v.name}.json`, OUT), JSON.stringify(tree));
    summary.push({ variant: v.name, pipelines: Object.keys(objects.pipelines).length, buffers: Object.keys(objects.buffers).length, nets: fh.nets });
    console.log(`${v.name}: ${Object.keys(objects.pipelines).length} pipelines, ${Object.keys(objects.buffers).length} buffers, nets ${JSON.stringify(fh.nets)}`);
  }
  let blobBytes = 0;
  for (const [h, data] of blobs) { writeFileSync(new URL(`blobs/${h}.bin`, OUT), data); blobBytes += data.byteLength; }
  console.log(`${blobs.size} blobs, ${(blobBytes / 1024).toFixed(0)} KB`);
  const compiled = {}, failed = [];
  for (const [h, code] of modules) {
    try { compiled[h] = compile(h, code); } catch (e) { if (!e.tool) throw e; failed.push({ h, message: e.message }); }
  }
  // ai: WebGPU's: the plain variant's GLSL kernels as WGSL (webgpuOf); the forms WGSL cannot hold listed, not failed
  const webgpu = { variant: WEBGPU_VARIANT, modules: [], not: [] };
  if (existsSync(TOOLS.spv2wgsl)) {
    for (const h of used[WEBGPU_VARIANT] ?? []) {
      const code = modules.get(h);
      if (code?.lang !== "glsl" || !compiled[h]) continue;
      try { webgpuOf(h, code.code); webgpu.modules.push(h); } catch (e) {
        if (!e.tool) throw e;
        const why = e.message.match(/: (parse|validate|write): (.*)/);
        webgpu.not.push({ h, message: why ? `${why[1]}: ${why[2]}` : e.message.split("\n")[0] });
      }
    }
    console.log(`webgpu (${WEBGPU_VARIANT}): ${webgpu.modules.length} modules as WGSL, ${webgpu.not.length} not (${[...new Set(webgpu.not.map((n) => n.message.slice(0, 70)))].join("; ")})`);
    packWebgpu(webgpu.modules);
  } else console.log(`webgpu: no ${TOOLS.spv2wgsl} (lizard-android/build.sh tools builds it): no WGSL written`);
  // ai: The sender's GPU encoder (gen/sender.mjs, 2026-10-02): its kernels as out/spv/send_<name>.spv, its manifest
  // ai: setup/send.json and its paint's TAB blobs/<hash>.bin (the codes and the whitening), for core/tx/gpu_painter.cpp.
  const send = await senderSetup();
  writeFileSync(new URL(`blobs/${send.tab.hash}.bin`, OUT), send.tab.bytes);
  for (const [k, code] of Object.entries(send.modules)) {
    try { compiled[`send_${k}`] = compile(`send_${k}`, { lang: "wgsl", code }); } catch (e) { if (!e.tool) throw e; failed.push({ h: `send_${k}`, message: e.message }); }
  }
  writeFileSync(new URL("setup/send.json", OUT), JSON.stringify(send.manifest));
  console.log(`sender: ${Object.keys(send.modules).length} kernels, TAB ${(send.tab.bytes.byteLength / 1024).toFixed(0)} KB`);
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
  // ai: What an earlier run wrote and this one did not (a module or a blob no setup names now) let go, so a tree
  // ai: copied whole (the desktop's bundle, an install) carries no stale file. Only gen's own names: a file put in
  // ai: spv/ by hand (bench_dot.spv) stays.
  const kept = new Set([...VARIANTS.map((v) => `setup/${v.name}.json`), "setup/send.json", ...[...blobs.keys(), send.tab.hash].map((h) => `blobs/${h}.bin`)]);
  for (const h of [...modules.keys(), ...Object.keys(send.modules).map((k) => `send_${k}`)]) for (const f of [`wgsl/${h}.wgsl`, `naga/${h}.wgsl`, `glsl/${h}.comp`, `spv/${h}.raw.spv`, `spv/${h}.spv`]) kept.add(f);
  for (const h of webgpu.modules) kept.add(`webgpu/${h}.wgsl`);
  for (const [dir, names] of [[ingestDir, "ingest_"], [txDir, "send_"]]) for (const f of readdirSync(dir).filter((f) => f.endsWith(".comp"))) kept.add(`spv/${names}${f.replace(/\.comp$/, "")}.spv`);
  const ours = { setup: /\.json$/, blobs: /^[0-9a-f]{16}\.bin$/, wgsl: /\.wgsl$/, naga: /\.wgsl$/, glsl: /\.comp$/, webgpu: /\.(wgsl|comp|spv)$/, spv: /^([0-9a-f]{16}|send_\w+|ingest_\w+)(\.raw)?\.spv$/ };
  let gone = 0;
  for (const [d, re] of Object.entries(ours)) for (const f of readdirSync(new URL(`${d}/`, OUT))) if (re.test(f) && !kept.has(`${d}/${f}`)) { rmSync(new URL(`${d}/${f}`, OUT)); gone++; }
  if (gone) console.log(`${gone} files of an earlier run let go`);
  // ai: the glslang that wrote the GLSL modules, by version: a module is named by its source, so another glslang writes
  // ai: other SPIR-V under the same names
  const glsl = Object.keys(compiled).filter((h) => compiled[h].glsl), glslang = run(TOOLS.glslang, ["--version"]).split("\n")[0].split(":").pop().trim();
  console.log(`${glsl.length} of them GLSL (glslang ${glslang})`);
  writeFileSync(new URL("gen.json", OUT), JSON.stringify({ sources: sourcesHash(), writer: `wgsl2spv (naga 30.0.1: zero-init native, no loop bounding, bound checks off); glslang ${glslang} (SPIR-V 1.3, no optimiser) for the GLSL modules`, variants: summary, modules: Object.keys(compiled), glsl, failed: failed.map((f) => f.h), webgpu }, null, 1));
  if (failed.length) process.exit(1);
}

if (MAIN) await main();
