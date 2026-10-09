# The GPU decoder

A second LIZARD decoder, designed from the format for a GPU. The C in `../src/` stays the reference and the
decoder where this one cannot run; this one is judged only on blocks that pass their CRC-32, on throughput, and on a
negative control. It has no parity with the C at any stage, not even on decisions.

What exists: the whole decode. The front half finds the symbol, orients it, identifies its ring (one of four
since 2026-09-27: 94, 158, 222 or 286 modules a side), registers it to a tenth of a module and reads its format word,
which names the picture (256 to 1536 samples, `../sim/lizard_pick.mjs` PICTURE_SIZES; any ring may carry any picture). The back half (`back/`, `back/DESIGN.md`) runs behind it in the batch's one
command encoder: the transform, whose first pass samples the n x n picture through the front half's map itself, with
the detrend, the soft values and the decline gate, the LDPC with the CRC-32. No grid is made (since 2026-09-24,
`back/DESIGN.md` 12); a batch reads back the verified blocks, a verdict a block and the counters (15 to 34 KB a frame,
against 4 MB for the grid). That is the default (`FrontHalf.create` `back: "gpu"`,
`scripts/exp/gpu_captures.mjs` and `scripts/exp/gpu_front.mjs` `BACK=gpu`). With `back: "wasm"` the grid is read back and the wasm back half (`sim/ob.mjs`
`Focus`, `focus_rx_finish`) scores it as a measuring instrument, which is what a page job naming no back half gets,
since the scripts sending those read the grid or the peaks; `BACK=both` does both on the same grids, frame by
frame. On the full replay the two back halves agree to 3 blocks in 93,000 (STATUS.md, 2026-09-24).

One symbol a frame. The dual code (two symbols in a 2:1 crop, `create({ symbols: 2 })`) was archived on 2026-09-28
and the archive deleted on 2026-09-29.

## Shape rules

- A batch is B frames as layers of one texture: one command encoder, one readback. Nothing in between waits on
  the host; later stages size themselves from counters earlier stages wrote.
- Every list has a fixed cap on the device. Hypotheses are enumerated and all scored; the frame's answer is an
  argmax. No early exit, no fallback chain, no retry. Every frame costs the same, and a miss is counted.
- Float throughout. Discrete choices are argmaxes over float scores.
- Explicit bind group layouts (never `layout: "auto"`); every function-scope `var` in WGSL gets an explicit
  initializer (Chrome on Vulkan kept a loop-body var's old value); a per-stage counter block comes back with
  every batch so a stage that did nothing reads as zeros; error scopes cover the whole batch.

## Stages (`wgsl/`)

| | file | what |
|---|---|---|
| F0 | ingest | only for a frame given as a VideoFrame (`../ingest.mjs`): its crop to luma, (77 R + 150 G + 29 B + 128) >> 8 on 8-bit values, in the frame's layer, a render pass a frame (r8unorm is not a storage format); a frame given as luma is written with writeTexture |
| levels | pyramid (one kernel with F1 since 2026-09-30) | not a decode stage: a 256-bin histogram of the middle half of each frame's crop, every second pixel each way, read back with the batch (`frames[i].hist`) so the receiver's grey range, median and clip shares come from the luma the decoder sees, a VideoFrame's included (no luma frame on the CPU under F0). Reads the layer, writes only its own buffer; the decode is untouched |
| F1 | pyramid | 2 x 2 means, levels 1 to 4 |
| F2 | bank | the peaks proposed to F4, each 16 x 16 tile of every level keeping its 4 strongest as (x, y, sigma, response). A trained fully convolutional proposer on levels 1 to 4 (`cnn/proposer/`), a module (`../bank_<name>.mjs`, contract in `wgsl/bank.mjs`): `fcn2` (version 2, the default since late 2026-09-24: each pixel standardised over its own 15 x 15 box, so the map is the level's alone and a workgroup takes a region of tiles; int8 where WGSL has it since 2026-09-26; `scripts/gpu/cnn/proposer/README.md` "Version 2") or `fcn` (version 1, the default until late 2026-09-24; `FCN_STORE=region=2` and `flat[=sd]` are its variants). The difference-of-Gaussians banks it replaced (`global`, `tiled`, `perscale`) were deleted on 2026-09-26 |
| F3 | bank SELECT | the strongest `cap` peaks a frame, one cut over every level |
| F4 | classify, RANK, classify | per peak: is it a corner mark, which way is out, how big a module. `classify` is the trained network (`cnn/`), run as a cascade: the small net (cnn-v3) on every kept peak, RANK keeps the 64 strongest by its mark logit (128 until 2026-09-24; STATUS.md, "Speed, round three") as a device-side list (a total order on the logit's bits, a NaN last; every list slot written each batch), the large net (cnn-v1) on those and overwrites their readings (`wgsl/classify.mjs CASCADE`; `CASCADE=off` runs one net on every peak). The kernel is `wgsl/classify_gemm.mjs` since 2026-09-24: several patches a workgroup, a register tile a lane, a split conv reduced by subgroup shuffles where the adapter has the `subgroups` feature (the decoder asks for it); `wgsl/classify.mjs` is the reference kernel `scripts/gpu/cnn/test_gemm.mjs` checks it against, and holds RANK and `CASCADE`. `DESCRIBE`, the hand-written score it replaced, was deleted on 2026-09-26 |
| F5 | finder VOTE, GATHER (PEAKS inside it since 2026-09-30) | every reading with p at least `SMIN` votes along a ray toward the symbol's centre (the diagonals cross there under any perspective), one lane a reading; the K strongest crossings each gather a quad in two rounds (the band about the crossing, then about the centre a filled diagonal pair gives); a quad is its four corners or nothing |
| F6 | finder SCORE, PICK | every (four-cornered quad, orientation, ring) read against the known timing track near the corners; argmax: the ring, never the picture |
| F7 | register NODES, REFIT | every border node correlated against the border as painted (256 lanes a node over offsets and sample groups, the coarse round on one shared lattice, the fine round moving each placed sample by the map's Jacobian); a homography and lens refit by Gauss-Newton as straight-line WGSL; each side's residual fitted by a polynomial of 1 to 6 coefficients, the count chosen by AICc on the side's own nodes (2026-09-29). Run twice; profiled as nodes1, refit1, nodes2, refit2 |
| F8 | word | the format word, one code a ring over every word cell (`src/fmt.h`: RS(8,3), (16,3), (24,3), (32,3) in the 32, 64, 96 and 128 rings), each cell read at its planned centre against its track pairs' grey, maximum likelihood over every version 1 to 128 and rate; the frame decodes at the word's version, or the batch's held one (the last word the host read) when none is taken, or not at all: `sel` = (ring, 1 + version, picture slot, 0), the block count and picture for the back half (after F7, on found and told frames alike); `scripts/gpu/word_check.mjs` holds its tables and decision to the wasm |
| F9 | sample | only when grids are read back (`BACK=wasm` or `both`): the picture through the map plus the Coons patch of the sides' residuals, in the strip order the back half reads. With the GPU back half B2's pass 1 samples instead, by the same map, patch (`sample.mjs COONS`, one copy) and bilinear read |
| B1 | back gate (`back_transform.mjs`) | lists each built picture's frames by the slot F8 gave them (a frame not valid, with no version, or naming an unbuilt picture is on no list) and writes the indirect dispatch sizes (pass 1, reduce and pass 2 a picture, and B4's first read: the most blocks and frames of a listed picture, so an empty slot costs nothing) and each frame's (w, h), blocks, rows and ring for the fused pass 1 (whose PIC holds each ring's grid for the picture and lattice); zeroes the counters, verdicts and record count, so the host clears nothing |
| B2 | back pass1, reduce, pass2 | pass 1 samples a column pair through the map (or reads F9's grid under `BACK=both`); the picture's 2D DFT kept as S = T / n^2 on the disc only: along y a column pair at a time (the column means apart, which keeps f16 safe), the detrend's five sums reduced, then along x a kept row at a time with the detrend subtracted in the spectrum; Y and S in f16 where the device has shader-f16 |
| B3 | back soft (`back_tiers.mjs` talign, tsoft, tblocks: the rate profile's stage, `back/tiers.mjs`, since 2026-10-07) | the grid's shift fitted from the pilots in the blocks' tails and turned back (since 2026-10-01); a sub-channel's 320 coefficients to int8 LLRs in slot order through their moments (`back_soft.mjs`'s arithmetic), with its pilot reading; a block's estimate over its own sub-channels (7, 8, 9 or 12 by its code), the block declined under its code's bar (0.2 k / n). Every version's profile is a row of the stage's table, so any version at any picture decodes |
| B4 | back ldpc (`back_tiers.mjs` tldpc, a dispatch a code: 7/8, 3/4, 2/3, 1/2) | a codeword a workgroup: bit map and whitening as one gather, layered min-sum in i16 with the C's norm and caps (`back_ldpc.mjs`'s), the bits past the payload and CRC known zeros, each code's own learned stop (a small net trained on that code's neutral codewords, `back/stop/`, `back/DESIGN.md` 6; `?ldpcstop=c` on the page, `LDPC_STOP=c` in the harness, is the C's stall rule), the CRC-32, and a verified block's 473 bytes compacted into the result record; one verdict and an iteration count a block |

`decoder.mjs` is the host; `tables.mjs` takes the format's tables from the wasm codec at start: a table a ring (the
border: lattice, kind map, marks, track and word plans) and the pictures (six since 2026-09-27), apart.

## Running

The harness and the measurement scripts named here (`scripts/`) are the lab's and not in this repository. The decoder
runs in the web receiver (`lizard-web/recv.html`, Decoder GPU or `?dec=gpu`) and natively through `liblizard/core`
(`lizard_gpu_check`, built by `lizard-android/build.sh gen` and `linux`).

Node has no WebGPU here, so `scripts/gpu/harness/run.mjs` spawns Chrome and trades jobs with `scripts/gpu/harness/page.mjs`. Plain runs
are SwiftShader (correctness only); `HWGPU=1` is the 4090 and `IGPU=1` the RDNA-2 iGPU, both needing
`DISPLAY=:0`. Only the iGPU's timings are reported (the 4090 counts blocks; its times say nothing about a phone). Wrap an iGPU timing in `scripts/tools/gpulock.sh time ...`, which waits for the machine to be quiet and runs alone, and everything else (4090 block counts, training, exports, SwiftShader) in `scripts/tools/gpulock.sh share ...`, which runs beside other shared jobs. The trained proposer (`bank_fcn.mjs`) takes 37.3 KB of workgroup memory in f32 (46 before the rewrite of
2026-09-24); on a device that gives less (SwiftShader and many phones give 32 KB) it keeps f32 arithmetic and stores
its activations as halves through `pack2x16float` (18.7 KB, as in f16; core WGSL, no `shader-f16`), and on a
device under that it throws `FitError`: no GPU decoder there, the C decodes (the global bank it fell back to until
2026-09-26 is deleted). `FCN_STORE=packed` forces the halves and `FCN_STORE=unfit` the throw, to measure either where
f32 fits. On the 4090 the halves gave 11,476 blocks on the 19 x 40 set against f32's 11,479 (the rewritten kernel and
the weights installed the evening of 2026-09-24; two runs of each, interleaved on one tree, each pair identical on
every frame; 34 frames moved by 1 or 2 blocks, 16 up and 18 down, no frame found or lost, bad 0). SwiftShader at the defaults 190 on 8 frames of it
(the C 184; 191 with the evening's weights) (2026-09-24). `bank_fcn2.mjs` takes the first form the device's workgroup
memory holds: f16 with a 32 x 32 region at 64 KB, f16 48 x 16 (31,340 B) at 32 KB, f32 32 x 16 on the 4090's 48 KB,
f32 16 x 16 on SwiftShader's 32 KB, and `FitError` below that. With `BACK=gpu` the replay counts apart the
found frames the back half's gate did not run (their size not built), rather than scoring them as frames that
decoded nothing. A real adapter's Chrome opens on the nested X display `:99`
(`scripts/tools/gpu_display.sh`, started by the harness when it is missing: Xvfb if installed, else an Xephyr whose one
window is hidden), so no window reaches the desktop; both cards are offered there unchanged, because Vulkan
enumerates them through the loader, not the X server. `GPU_DISPLAY=<display>` overrides.

```
HWGPU=1 node scripts/exp/gpu_front.mjs [frames]      simulator cells: ref, oracle, truth, find
HWGPU=1 node scripts/exp/gpu_rings.mjs               every (ring, picture) pair read told nothing, and the held word
HWGPU=1 node scripts/exp/gpu_captures.mjs            the recorded phone runs beside the C decoder (research/captures/v0.3)
HWGPU=1 node scripts/exp/gpu_marks.mjs                           does the bank keep the true marks among its peaks
```

`scripts/exp/gpu_captures.mjs` and `scripts/exp/gpu_front.mjs` take `BACK=gpu|wasm|both` (which back half makes the blocks, default gpu,
`scripts/gpu/harness/run.mjs` `backFromEnv`); `gpu_captures` also takes `FRAMES`, `RUNS=<substring>`, `B`, `PROFILE=1` (a timestamp pair a stage, the back half's six
too) and `PERFRAME=<path>` (one line a frame; `archive/build-scratch-2026-09/cnn_deficit.mjs` splits the deficit against the C from it). All three
scripts take `BANK=fcn2|fcn` (default `fcn2`, as in `FrontHalf.create` and `scripts/gpu/harness/page.mjs`), `WEIGHTS=<file under gpu/cnn/>`,
`PRECISION=f16` (AMD adapters only in Chrome) and `CAP`. The cascade is on by default; `CASCADE=off` runs the one net from `WEIGHTS` on every
peak, `CASCADE=<file>` names another small net, `CASCADE_KEEP=<n>` and `CASCADE_REST=small|zero` change its list
and what a peak off the list keeps (`gpu_captures` and `gpu_front`). Both take `B=auto|<n>` (default auto, the
batch controller below; a number fixes the batch as before), `BUDGET=<MB>` (default 128) and `BATCH_MS=<ms>`
(default 400) for it, and `LOSEAFTER=<n>` (the page destroys its device after the n-th batch, to test recovery).

### Batch size: `B=auto`

`FrontHalf.create`'s `B` is `"auto"` by default, and so is the harness's (a job naming no `B` still gets 8 from
`scripts/gpu/harness/page.mjs`). WebGPU tells a page its buffer and workgroup limits, not the memory free, the clocks or how long
a submit may run before the device is reset, so the controller (`decoder.mjs` `Batcher`) measures them on the
decoder's own batches.

- Memory. The first batch prices a frame: a one-frame lane at that batch's frame and picture size, its back half
  lane's bytes shared out a frame (the back half is built at create for `B_CEIL`, so the lane holds 32 frames of it;
  a one-frame back half, compiled for the purpose, before 2026-09-29). Then two lanes of Bmax frames are made, the largest batch whose lanes fit the budget beside
  what is shared (`budget`, the device's own `maxBufferSize` since 2026-09-25 (`budgetFor`; 2 GiB on the S26 Ultra, 4 on the 4090, 256 MiB on SwiftShader), WebGPU stating no memory size; the degrading is `makeLanes`: an allocation that fails its out-of-memory scope halves the batch down to 1; and never more than `B_CEIL` = 32 a batch (16 until 2026-09-25 10:44Z: as many as the camera's buffers allow or 16, whichever is lower); 128 MiB before, a budget of about 100 MB with margin), capped by the device's buffer
  size and texture layer limits; an allocation that fails is tried again at half. The back half's shaders are
  compiled for a batch size: at create for `B_CEIL`, and again in planning only for lanes memory holds under it
  (SwiftShader). A plan reads only the frames' shapes (3 to 14 ms on the iGPU and the 4090, 2026-09-29).
- Time. The stream starts at Bmax (since 2026-09-29). Until then the first plan
  doubled a batch of copies of its first frame from 1 while its device time held under `capMs` (400), and settled on
  the largest that held: 1.34 s on the iGPU (the back half compiled at 1 and at 32 about 1 s of it), and on the S26
  2.6 s in which the camera delivered almost nothing (STATUS "The ramp at the start"). 400 ms is a fifth of the
  submit of about 2 s that lost the iGPU's device, which leaves room for throttling to double a batch's time, and
  for a batch dearer than the ones before it, before the watch shrinks it.
- Watch. Every batch but a lane's first (every pipeline's and buffer's first use; `warm()` ran one frame through
  each lane for it before 2026-09-29): one over the cap halves the size at once, three running over 90% of the cap
  halve it, sixteen running with room for a double grow it back toward Bmax. A batch with a larger frame or picture
  than the lanes were planned for plans them again, once every lane is idle.
- In flight: 2 lanes. A trial of 3 in the same budget (2026-09-25) never fired on the iGPU (96 to 99% busy at 2)
  and on the phone measured a starved feed, not the device, so it went.
- The ring (the GPU queue, `decoder.mjs` `enqueue`): Bmax + 4 layers at the lanes' shape, counted in the budget
  beside the lanes, so Bmax is a little smaller with it where memory binds (SwiftShader; on the S26, the iGPU and the
  4090 `B_CEIL` binds first since the budget became the device's own `maxBufferSize`). A frame
  goes onto it the moment it arrives and its VideoFrame is closed; a batch is cut from staged slots, its encoder
  beginning with a copy a slot into the lane's layer, and the slots are freed at the submit.
- Device lost. The page makes a new device and a decoder that starts at half the batch that was running and never
  grows past it (`create` `restart`), and runs the lost batches again on it, in order.
- A batch carries any number of frames up to its lanes' B. Every dispatch covers only the frames it carries (the
  back half's soft values and LDPC too, `back.mjs` `encode` `frames`), clears and readbacks cover only them, and an
  empty slot exits at once; the back half's gate still walks every slot in one thread. On the iGPU a batch of 1 in
  lanes of 111 took 10.8 ms against 10.0 in lanes of 1 (11.9 while the soft values and the LDPC still dispatched every
  slot, 2026-09-24), most likely lane 1's one-time zero-fill on its first use rather than the gate (STATUS.md,
  "Dynamic batching").
- The page asks node for jobs of the controller's size (`GET /job?want=n`, `scripts/gpu/harness/run.mjs` `autoJobs`) and cuts
  a larger one (the first, asked before the plan, or one asked before a change) into batches of the size wanted as
  each goes; its batches' results go back as one. The total line names the size, Bmax, the limit that bound it, the
  in-flight count and why, and the plan's time, which comes before the first submit and so outside the span.

On the iGPU auto takes `B_CEIL`'s 32 at 1080 since the budget became the device's `maxBufferSize` (2026-09-25), and
the 400 ms cap holds 2160 to about 8. At the 128 MiB budget before, it took 13 frames at 1080 and 4 at 2160, both
bound by memory, and its rate matched the best fixed B; no batch past 8 at 1080 or 4 at 2160 bought throughput, since
the device was 96 to 98% busy at every B (STATUS.md, "Dynamic batching", 2026-09-24).

### Batches in flight, and what the total line means

`INFLIGHT=<n>` (default auto: 2) is how many batches the page keeps on the device at once. `FrontHalf.create`'s
`inflight` makes that many lanes (a batch's working buffers, bind groups, timestamps and readback; pipelines,
tables and weights are compiled and uploaded once and shared), `run()` takes a free one and returns as soon as
the batch's frames are on the ring and the batch is submitted, its promise being the readback, and the page submits
the next batch while the ones before it run. A batch's readback is requested at its submit, so it never waits on a later submit. The
blocks are the same at any n; device memory is n times a lane (a 1080 batch of 8 with a 1024 picture is 37 MB a lane
on the iGPU in f16 and 46 MB on the 4090 in f32 with the GPU back half, which makes no grid; about 89 MB with the wasm back
half, two thirds of it the grid and its readback); a submit is still one batch, so n does not lengthen
it (the iGPU loses its device on a submit over about 2 s; the 2160 run at B=8 was 2.3 s there and is 0.22 s
now, and its rows in STATUS.md stay at B=2 for comparison).

The total line is the throughput. Frames a second is the page's span, first submit to last readback, over the
real frames (empty slots count as nothing), each run's span summed; the steady rate beside it leaves out the
first batch (its frames and the wait for its readback), so it is the pipe once full. Compute ms a frame is the
stage timestamps, and the device busy share is the two multiplied: near 100% the device bounds the run (the
iGPU at n = 2), far below it the host does. What the page waited for between batches says which host part: a
job still on its way from node (the HTTP trade, about 9 MB of frames in a batch of 8 at 1080, and back 33 MB of
f32 grids with the wasm back half or 60 to 270 KB of blocks and counters with the GPU's; on the 4090 the trade, not
the device, bounds a run) or a lane still busy (the device). `PREFETCH=<n>` (default 2) is how many jobs the page
keeps on their way from node at once; it moves only the transport-bound 4090. A result's body is one buffer (33 MB a
batch of 8 at 1080 with a 1024 picture), not a Blob: a Blob's bytes stay in Chrome's blob store until its wrapper is
collected, and on a 600-frame run on the 4090 a post failed ("Failed to fetch"). The page also keeps at most
four results on their way back to node, and a wait for a post is the third wait on the total line: a long 4090
run is bound there at any n in flight, so its rate is the transport's, not the decoder's. Latency a batch is its upload to its readback, which n in flight lengthens; it is
not the number that matters. Under `PROFILE=1` a run also prints the page's side of a batch in ms: upload (the
frames into the texture), encode, scopes (the error scopes' round trip), map (the readback's wait), read (out
of the mapped range), a job's fetch, and the two waits.
