# core/nets: native kernels on the cooperative-matrix units

What is here (2026-09-30):
- `cls.glsl`, `cls_v3.comp` with `../../gen/nets_cls.mjs`: the small classifier (cnn-v3) as an int8 kernel on the
  matrix units, exact, 0.356 ms a frame on the S26 against the web's int8 kernel's 0.479 (3.02 to 3.04 against
  3.08 ms of GPU a frame; through the host's guard at thermal status 0 with the app stopped: f16 0.776, int8 0.476,
  native 0.358, the native kept, 2,390 blocks). Not in the manifest by default (`LIZ_NETS_CLS=1`): it adds 1.5 to 1.8 s to a cold start and leans on a
  compiler with the faults listed below. Where it is listed, the host adopts it only after its first-batch check
  (`../dec/front.cpp` measureTwins: the web int8 kernel's outputs bit for bit, and the fastest form).
- `bench_coop.comp`, `bench_dot.comp`, `bench_h16.comp`: the multiply-add microbenchmarks (`lizard_gpu_check bench`).
- Set aside: the large classifier (cnn-v1: its f16 form already costs 0.108 ms a frame) and the proposer's network
  (`archive/stash/coopmat-proposer/`: exact, 5.4 times slower than the web's kernel).

## State at the end (07:15)

- The kernel: `core/nets/cls.glsl` (compile unit `cls_v3.comp`), generator `gen/nets_cls.mjs`. cnn-v3 only; cnn-v1
  dropped (its f16 form costs 0.108 ms a frame), `cls_v1.comp` deleted. One arm a build, chosen by the FAST
  specialisation constant: FAST 1 the fast build (a lane writes its GEMM row into A's elements and reads its C row's
  elements; the Adreno), FAST 0 the staging build (rows and accumulators through shared memory; any device; the
  4090's check).
- Manifest entry to ship (what `LIZ_NETS_CLS=1 node gen/nets.mjs <out>` writes; without the variable `build()`
  returns no entry):
  `{"variants":["int8-sg","int8"],"role":"front.twins.small.pipe","weights":"front.twins.small.buf","spv":"nets/cls_v3.spv","spec":[64,32,1,0,0,4],"P":4,"blob":"nets/cls_v3.bin","needs":[{"M":64,"N":32,"K":32,"A":3,"B":3,"C":5,"R":5},{"M":64,"N":16,"K":32,"A":3,"B":3,"C":5,"R":5}]}`
  spec = [M, NW, FAST, STOP, DBG, PP]; the stride unit is the build's -DSU (adreno 4u, nv16 1u). The 4090 check
  build is `LIZ_NETS_SHAPES=nv16` with spec [16,16,0,0,0,4]. P is PP again, for the host: it dispatches
  ceil(slots / P) workgroups a frame (2026-09-30; a check that edits spec's PP edits P with it, or the slots past
  ceil(slots / P) PP are never read).
- What it writes, and the WGSL int8 twin writes (`out/wgsl/87f843a78486af55.wgsl`, `c897ccd76052bc1a.wgsl`), for each
  slot i under the frame's count n = min(counts[16 f + 2], cap), and nothing for a slot past it or a frame whose
  `valid` is 0: `score[f cap + i]` (one f32, the mark logit z0) and the reading's three vec4 at `readings[3 (f cap +
  i)]`, all twelve words: (peak x, peak y, u exp(z3), atan(z2, z1) - pi / 4), (1 / (1 + exp(-z0)), 0, 0, 0),
  (z4 > 0 ? 1 : 0, peak w, level, 1). No other buffer. With DBG 0 (the shipped spec) the thirteen words equal the
  twin's to the bit on the phone (the compare tool compares the u32 words). DBG 1 puts the path and the stride unit in
  the second vec4's y and z: never in a shipped spec. A fast build whose layout check fails writes the same words as
  "no mark": z0 = -1e30 (score -1e30, probability 0), z1 = 1, the rest 0, so an exact-equality guard rejects it.
- Exact (the compare tool, 8 frames):
  - phone, fast build: "8192 peaks, 0 not in both runs; readings differ: the large net's 0 of 512, the small net's 0 of
    7680 (and 0 of 0 listed in one run only); logits differ on 0". DBG 1: path 1 on all 7,677 small readings.
  - 4090, staging build: "8192 peaks, 15 not in both runs; readings differ: the large net's 0 of 511, the small net's 0
    of 7665 (and 1 of 1 listed in one run only); logits differ on 0", 251 blocks. DBG 1: path 2, stride unit 1.
  - phone, staging build (FAST 0): path 2, unit 4 probed, readings equal on every peak, 226 blocks.
  - phone, a fast build with the wrong unit (-DSU=1u): path 3 (no mark) on all 7,680, 0 blocks: nothing, not garbage.
- Blocks on the phone, with the kernel and without: selftest 226 and 226; replay run-2026-09-27T07-56-38-254Z 96
  frames 2,390 and 2,390 (LIZ_TWINS=int8, and the default run, which keeps the kernel).
- Speed on the phone, the decoder's own twin line (`int8 twins measured on 2 frames`, ms a frame of 1,024 patches),
  07:14:38 to 07:14:46, thermal status 0 before and after every run, `native net: front.twins.small.pipe` in every
  run's log, the app in front but idle on the GPU (so provisional):
  - the fast build: small int8 0.356, 0.356, 0.359 (f16 0.770, 0.779, 0.761; int8 kept each time), 2,390 verified
    blocks, 3.04 ms of GPU a frame in each.
  - the WGSL int8 twin in the same minute (LIZ_NATIVE_NETS=0): int8 0.479 (f16 0.766), 2,390 blocks, 3.08 ms of GPU a
    frame. So the kernel is 0.12 ms a frame (26%) under the twin it replaces.
  - the fast build by STOP: the cut 0.096, + conv1 0.138, + conv2 0.281, whole 0.356 (STOP 4 does not compile).
  - at thermal status 1 earlier (07:08 to 07:11, the GPU unthrottled): 0.355, 0.356, 0.355; STOP 3 0.300; the staging
    build on the phone 3.20 (f16 kept). Both arms in one shader: 4.35 to 4.57 (status 0, 06:42 to 06:46).
- Setup build time on the phone (`selftest`, thermal status 1, two rounds): cold (cache_cls.bin removed) 3.20 and
  3.49 s without the kernel, 4.96 and 4.99 s with it: the kernel adds 1.5 to 1.8 s to a cold start. Warm 36 to 44 ms
  without, 38 to 42 ms with. The kernel alone against a warm cache: 1.19 to 1.50 s.
- Open: the timings with the app not in front; which instruction of the staging arm slows the fast arm when both are in one
  shader (fault 2 below); the cold compile's 1.5 s (the fast arm's unrolled row builders and element accesses; not
  tried in loops); the staging build is a check path on the phone (3.2 ms), not a fallback the host can use.

## The Adreno 840 compiler and cooperative matrices: faults, what triggers each, what avoids it

Driver "Adreno Vulkan Driver Build 87ff20b216, Ifbe74a3179", compiler E031.50.19.18, Vulkan 1.4.295 (SM-S948U1).
1. The stride unit. `coopMatLoad` and `coopMatStore` count `stride` in the matrix's component size; SPIR-V
   (SPV_KHR_cooperative_matrix, issue 3) counts it in elements of the array pointed into, and the 4090 does that. An
   int8 matrix in a `uint` array with stride s: rows 4 s bytes apart on the 4090, s bytes apart on the Adreno. int32
   matrices in a `uint` array agree. Avoids it: give the stride in the device's unit (here -DSU 4u for the Adreno, and
   the fast build's layout check tests it), or probe it (the staging build's `strideUnit`: a store of a matrix whose
   rows are bytes 0..31 with stride 8; byte 32 M - 1 is written only where a stride counts words), or load from an
   array of the component's own type.
2. Two arms in one shader. A kernel that holds both the element arm and the staging arm (a workgroup-uniform run-time
   bool between them) runs the element arm 12 times slower: 4.35 to 4.57 ms against 0.355, conv2's block 3.3 ms
   against 0.16. Neither literal strides in both arms (describe 6.13 against 6.44 to 6.63) nor a specialisation-constant
   stride changed it. Avoids it: one arm a shader, chosen at specialisation (the driver drops the other). Not bisected
   to an instruction (the staging arm's coopMatStore and accumulator loads through shared memory, or the branch).
3. vkCreateComputePipelines fails with VkResult -13 when a matrix value is consumed more than once: one A in three
   multiply-adds; two or three multiply-adds each taking `C16_t(0)` as the accumulator (whatever the strides, with
   or without a barrier between them); `c0 - c0` on accumulators. Avoids it: every matrix value used once; a second
   zero accumulator from a stride-0 coopMatLoad of zeroed memory. The fast build's STOP 4 (conv3 with fc1 compiled
   out) fails the same way; STOP 0 to 3 compile.
4. "error: ran out of registers during register allocation" (the pipeline is refused): cnn-v1's element arm forced
   (32-wide accumulators, three K steps, C read by elements); two accumulators loaded from the weights buffer with
   stride 0 plus five B matrices held across the patch loop. An accumulator is 16 or 32 registers a lane. Avoids it:
   fewer matrices alive at once.
5. Wrong values with no error, in cnn-v1's two-arm kernel only: an int8 matrix loaded from shared memory with stride 0
   read words 2 and 3 of each 16 bytes as words 0 and 1 (elements 8..15 = 0..7, 24..31 = 16..23), and both the
   A-through-memory form and the staging arm gave wrong readings there (32 and 61 blocks of 226). cnn-v3's builds do
   not show it. Avoids it: nothing known but comparing readings on the phone after every structural change.
6. Readings that fault 2 confounded (every EXP shader of sections 7 and 8 held conv1's staging arm): "a stride
   that is not a literal costs milliseconds" and "straight-line per-lane code after matrix instructions is
   pathological". Both were real in those shaders (conv2 0.78 with a literal stride against 3.07 with a run-time one;
   16 unrolled loads 2.15 against 0.27 looped) and neither has been measured in a one-arm shader. The fast build is
   all unrolled element accesses at constant indices with literal strides, and is fast and exact.
7. Seen in the proposer's kernel (not reproduced here): a per-lane register assigned under a per-lane condition and live
   across a barrier can keep its old value in a module with matrix instructions; a matrix element read at a constant
   index crashed its compile (this kernel's fast arm reads and writes elements at constant indices and compiles).

## Cost models, for the record (thermal status 0, 06:42 to 06:46; no staging arm in either)
- Memory-fed (every A one coopMatLoad at a literal stride, conv1's and conv2's B loaded once a workgroup, every
  accumulator stored with coopMatStore and requantised from memory in a loop; the right op counts, wrong rows):
  twin line 1.35, describe 3.71 and 3.70 (the twins 2.63 to 2.79).
- Hybrid (the same A side, every accumulator requantised from its elements in a loop of four): twin line 0.31,
  describe 2.58. Not written exactly (06:50): the existing kernel reached 0.355 first.
- Their sources and the EXP arms are not in this repository.

## How to build and check
- Build: `LIZ_NETS_CLS=1 node liblizard/gen/nets.mjs <tree>` (adreno), `LIZ_NETS_CLS=1 LIZ_NETS_SHAPES=nv16 node liblizard/gen/nets.mjs
  <tree>` (the 4090's staging build); `node liblizard/gen/nets_cls.mjs check` (the packed GEMM against the contract, no GPU).
  Both builds need glslang 16, placed by hand at `liblizard/.tools/glslang/bin/glslang` (a KhronosGroup/glslang release).
- The bench kernels: `lizard_gpu_check bench` reads `<LIZ_OUT>/spv/bench_dot.spv`, and `bench_coop.spv` and `bench_h16.spv` where the
  device has cooperative matrices; no build step makes them. From the root, with the same glslang (this gives the
  bytes the bench was run on, 2026-09-29):
  `for k in coop dot h16; do liblizard/.tools/glslang/bin/glslang -V --target-env vulkan1.3 liblizard/core/nets/bench_$k.comp -o liblizard/out/spv/bench_$k.spv; done`.
- A check edits the manifest's spec array: DBG 1 for the path, FAST 0 for staging, STOP n for a profile.
- Phone: `PULL=<dir> lizard-android/tools/phone/net.sh cls <nets dir> LIZ_TWINS=int8 LIZ_NATIVE_NETS=1 -- dump int8-sg d_cls`, then
  `node liblizard/gen/nets_cls.mjs compare <reference dump> <dir> 8`; the twin line from `lizard-android/tools/phone/net.sh cls <nets dir>
  LIZ_NATIVE_NETS=1 -- replay run-2026-09-27T07-56-38-254Z 96 int8-sg`.
- 4090: `LIZ_OUT=<nv16 tree> LIZ_TWINS=int8 LIZ_NATIVE_NETS=1 LIZ_VK_DEVICE=4090 scripts/tools/gpulock.sh share
  lizard-android/build/linux/lizard_gpu_check dump int8-sg <dir>` against the same with `LIZ_NATIVE_NETS=0`.
