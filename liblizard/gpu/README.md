# The GPU decoder

A second LIZARD decoder, designed from the format for a GPU. The C in `../src/` stays the reference and the
decoder where this one cannot run; this one is judged only on blocks that pass their CRC-32, on throughput, and on a
negative control. It has no parity with the C at any stage, not even on decisions.

What exists: the whole decode. The front half finds the symbol, orients it, identifies its ring (one of three: 158,
286 or 542 modules a side), registers it, reads the anchors over the picture and reads its format word, which names
the picture (256 to 1536 samples, `../sim/lizard_pick.mjs` PICTURE_SIZES; any ring may carry any picture). The back
half (`back/`, `back/DESIGN.md`) runs behind it in the batch's one command encoder: the transform, whose first pass
samples the n x n picture through the front half's map itself, with the detrend, the soft values and the decline
gate, the LDPC with the CRC-32. No grid is made (`back/DESIGN.md` 10); a batch reads back the verified blocks, a
verdict a block and the counters. That is the default (`FrontHalf.create` `back: "gpu"`, `scripts/exp/gpu_captures.mjs`
and `scripts/exp/gpu_front.mjs` `BACK=gpu`). With `back: "wasm"` the grid is read back and the wasm back half
(`sim/ob.mjs` `Focus`, `focus_rx_finish`) scores it as a measuring instrument, which is what a page job naming no back
half gets, since the scripts sending those read the grid or the peaks.

One symbol a frame; a 2:1 capture is two frames, a half each.

## Shape rules

- A batch is B frames as layers of one texture: one command encoder, one readback. Nothing in between waits on
  the host; later stages size themselves from counters earlier stages wrote.
- Every list has a fixed cap on the device. Hypotheses are enumerated and all scored; the frame's answer is an
  argmax. The finder runs twice at most: its first round on the proposer's levels 2 to 4, a second round with level 1
  on the frames the first found nothing on (`wgsl/finder.mjs` REDO); past that, no early exit, no fallback chain, no
  retry, and a miss is counted.
- Float throughout. Discrete choices are argmaxes over float scores.
- Explicit bind group layouts (never `layout: "auto"`); every function-scope `var` in WGSL gets an explicit
  initializer (Chrome on Vulkan kept a loop-body var's old value); a per-stage counter block comes back with
  every batch so a stage that did nothing reads as zeros; error scopes cover the whole batch.

## Stages (`wgsl/`)

| | file | what |
|---|---|---|
| levels | pyramid (one kernel with F1) | not a decode stage: a 256-bin histogram of the middle half of each frame's crop, every second pixel each way, read back with the batch (`frames[i].hist`) so the receiver's grey range, median and clip shares come from the luma the decoder sees, a native camera frame's included (its luma never reaches the CPU). Reads the layer, writes only its own buffer; the decode is untouched |
| F1 | pyramid | 2 x 2 means, levels 1 to 4 |
| F2 | bank | the peaks proposed to F4, each 16 x 16 tile of every level keeping its 4 strongest as (x, y, sigma, response). The trained fully convolutional proposer on levels 1 to 4 (`cnn/proposer/`), hosted by `bank_fcn2.mjs` to the contract in `wgsl/bank.mjs`: each pixel standardised over its own 15 x 15 box, so the map is the level's alone and a workgroup takes a region of tiles; int8 where WGSL has it (`scripts/gpu/cnn/proposer/README.md` "Version 2") |
| F3 | bank SELECT | the strongest `cap` peaks a frame, one cut over every level |
| F4 | classify | per peak: is it a corner mark, which way is out, how big a module. `classify` is the trained network (`cnn/`, cnn-v3), run on every kept peak. The kernel is `wgsl/classify_gemm.mjs`: several patches a workgroup, a register tile a lane, a split conv reduced by subgroup shuffles where the adapter has the `subgroups` feature (the decoder asks for it); `wgsl/classify.mjs` is the reference kernel `scripts/gpu/cnn/test_gemm.mjs` checks it against, and names the installed weights (`CLASSIFIER`) |
| F5 | finder VOTE, GATHER (PEAKS inside it) | every reading with p at least `SMIN` votes along a ray toward the symbol's centre (the diagonals cross there under any perspective), one lane a reading; the K strongest crossings each gather a quad in two rounds (the band about the crossing, then about the centre a filled diagonal pair gives); a quad is its four corners or nothing |
| F5d | finder COMPLETE (`wgsl/finder.mjs`) | a quad of three corners where no quad of four was gathered: the fourth from the crossing of the ring's two sides read toward it (the corner a crop or the nets missed), held to SCORE as any quad |
| F6 | finder SCORE, PICK | every (four-cornered quad, orientation, ring) read against the known timing track near the corners; argmax: the ring, never the picture |
| F7 | register NODES, REFIT, REFIT_POST | every border node correlated against the border as painted (256 lanes a node over offsets and sample groups, +-3 modules at half a module on one shared lattice about the finder's map); REFIT a homography by Gauss-Newton as straight-line WGSL, then REFIT_POST the lens about the camera's axis and each side's residual fitted by a polynomial of 1 to 6 coefficients, the count chosen by AICc on the side's own nodes. One round (the anchors' stages search the border again about their own model, F7a); profiled as nodes and refit |
| F7a | anchors ANCH_FIND, ANCH_PREDICT, ANCH_REFIT, ANCH_FIT (`wgsl/anchors.mjs`; every picture carries them, SPEC 6.13) | the 7 x 7 grid of anchor squares: after F7, each square found in a window one side of it wide each way about where the frame's final map puts it, by the pattern's normalised correlation at every step (box sums of the window's integral images, a parabola each way); on a frame with 12 found or more (`ANCH_MIN`), ANCH_PREDICT fits the lens over the anchors found and predicts each border node from it, and the border is searched again about each prediction, +-0.4 module (F7's node search, centred; a frame without the anchors keeps F7's nodes); ANCH_REFIT fits the lens about the camera's axis again over the border's nodes as measured and the anchors (K radial terms by AICc with no cap, the anchors sampling the interior's radii; robust over the border: a node further from the model than half a module or three robust spreads is dropped and reads the model, four fits at most), each side's curve of what it leaves on the border, and each anchor's residual against that map (`../src/acquire.c` `border_fill_anchors` is the same fit); then ANCH_FIT takes a frame's anchors' residuals whole and fills the missing ones from their neighbours, with the picture's mean, into the frame's record (`result` word 31 the anchors found). A told batch, or one with registration off, runs ANCH_FIND and ANCH_FIT alone. The back half's pass 1 then takes a cubic mesh of those residuals into every sample's position and sets the squares to the mean before the detrend. F9 reads no anchors |
| F8 | word | the format word, one code a ring over every word cell (`src/fmt.h`: RS(16,3), (32,3), (64,3) in the 64, 128 and 256 rings), each cell read at its planned centre against its track pairs' grey, ordered statistics decoding at order 3 over versions 2 to 128 and every rate, taken by a side-aware gate at the C's false-word rate (`wordcode.mjs` `gateTable`); the frame decodes at the word's version, or the batch's held one (the last word the host read) when none is taken, or not at all: `sel` = (ring, 1 + version, picture slot, 0), the block count and picture for the back half (after F7, on found and told frames alike); `scripts/gpu/word_check.mjs` holds its tables and decision to the wasm |
| F9 | sample | only for the wasm back half (`back: "wasm"`, `BACK=wasm`), whose grids are read back: the picture through the map plus the Coons patch of the sides' residuals, in the strip order the back half reads. With the GPU back half B2's pass 1 samples instead, by the same map, patch (`sample.mjs COONS`, one copy) and bilinear read |
| B1 | back gate (`back_transform.mjs`) | lists each built picture's frames by the slot F8 gave them (a frame not valid, with no version, or naming an unbuilt picture is on no list) and writes the indirect dispatch sizes (pass 1, reduce and pass 2 a picture) and each frame's (w, h), blocks, rows and ring for the fused pass 1 (whose PIC holds each ring's grid for the picture and lattice); zeroes the counters, verdicts and record count, so the host clears nothing |
| B2 | back pass1, reduce, pass2 | pass 1 samples a column pair through the map; the picture's 2D DFT kept as S = T / n^2 on the disc only: along y a column pair at a time (the column means apart, which keeps f16 safe), the detrend's five sums reduced, then along x a kept row at a time with the detrend subtracted in the spectrum; Y and S in f16 where the device has shader-f16 |
| B3 | back soft (`back_tiers.mjs` talign, tsoft, tblocks: the rate profile's stage, `back/tiers.mjs`) | the grid's shift fitted from the pilots in the blocks' tails and turned back; a sub-channel's 320 coefficients to int8 LLRs in slot order through their moments (the C's arithmetic, `src/focus.c`), with its pilot reading; a block's estimate over its own sub-channels (7, 8, 9 or 12 by its code), the block declined under its code's bar (0.2 k / n). Every version's profile is a row of the stage's table, so any version at any picture decodes |
| B4 | back ldpc (`back_tiers.mjs` tldpc, a dispatch a code: 7/8, 3/4, 2/3, 1/2) | a codeword a workgroup: bit map and whitening as one gather, layered min-sum in i16 with the C's norm and caps (`src/ldpc.c`), the bits past the payload and CRC known zeros, each code's own learned stop (a small net trained on that code's neutral codewords, `back/stop/`, `back/DESIGN.md` 6), the CRC-32, and a verified block's 473 bytes compacted into the result record; one verdict and an iteration count a block |

`decoder.mjs` is the host; `tables.mjs` takes the format's tables from the wasm codec at start: a table a ring (the
border: lattice, kind map, marks, track and word plans) and the six pictures, apart.

## Running

The harness and the measurement scripts named here (`scripts/`) are the lab's and not in this repository. The decoder
runs natively through `liblizard/core` (the Android and desktop apps' receivers, and `lizard_gpu_check`): this JS host
builds that setup on a recording device (`liblizard/gen/gen.mjs`, `lizard-android/build.sh gen`), its WGSL compiled to
SPIR-V, and runs on WebGPU only in the lab's harness. The web receiver decodes with the C (`../src/`); the web sender
keeps the WebGPU encoder (`encoder.mjs`).

Node has no WebGPU here, so `scripts/gpu/harness/run.mjs` spawns Chrome and trades jobs with `scripts/gpu/harness/page.mjs`. Plain runs
are SwiftShader (correctness only); `HWGPU=1` is the 4090 and `IGPU=1` the RDNA-2 iGPU, both needing
`DISPLAY=:0`. Only the iGPU's timings are reported (the 4090 counts blocks; its times say nothing about a phone). Wrap an iGPU timing in `scripts/tools/gpulock.sh time ...`, which waits for the machine to be quiet and runs alone, and everything else (4090 block counts, training, exports, SwiftShader) in `scripts/tools/gpulock.sh share ...`, which runs beside other shared jobs. `bank_fcn2.mjs` takes the first form the device's workgroup
memory holds: f16 with a 32 x 32 region at 64 KB, f16 48 x 16 (31,340 B) at 32 KB, f32 32 x 16 on the 4090's 48 KB,
f32 16 x 16 on SwiftShader's 32 KB, f32 arithmetic with its activations stored as halves through `pack2x16float` on a
device without `shader-f16` that holds no f32 form, and `FitError` below that: no GPU decoder there, the C decodes.
`FCN_STORE=` (the page's `?fcnstore=`) runs a form a smaller device would be given where a larger one fits: `packed`
the halves, `f32`, `region=32x16`, `fused`, `xblock=32x32`, and `unfit` the throw. With `BACK=gpu` the replay counts apart the
found frames the back half's gate did not run (their size not built), rather than scoring them as frames that
decoded nothing. A real adapter's Chrome opens on the nested X display `:99`
(`scripts/tools/gpu_display.sh`, started by the harness when it is missing: Xvfb if installed, else an Xephyr whose one
window is hidden), so no window reaches the desktop; both cards are offered there unchanged, because Vulkan
enumerates them through the loader, not the X server. `GPU_DISPLAY=<display>` overrides.

```
HWGPU=1 node scripts/exp/gpu_front.mjs [frames]      simulator cells: ref, oracle, truth, find
HWGPU=1 node scripts/exp/gpu_rings.mjs               every (ring, picture) pair read told nothing, and the held word
HWGPU=1 node scripts/exp/gpu_captures.mjs            the replays (research/replays) beside the C decoder
HWGPU=1 node scripts/exp/gpu_marks.mjs                           does the bank keep the true marks among its peaks
```

`scripts/exp/gpu_captures.mjs` and `scripts/exp/gpu_front.mjs` take `BACK=gpu|wasm` (which back half makes the blocks, default gpu,
`scripts/gpu/harness/run.mjs` `backFromEnv`); `gpu_captures` also takes `FRAMES`, `RUNS=<substring>`, `B`, `PROFILE=1` (a timestamp pair a stage, the back half's six
too) and `PERFRAME=<path>` (one line a frame). All three
scripts take `WEIGHTS=<file under gpu/cnn/>`,
`PRECISION=f16` (AMD adapters only in Chrome) and `CAP`. One net runs on every peak (`WEIGHTS`, the installed cnn-v3 by default).
Both take `B=auto|<n>` (default auto, the batch controller below; a number fixes the batch),
`BUDGET=<MB>` (default the device's own `maxBufferSize`) and `BATCH_MS=<ms>`
(default 400) for it, and `LOSEAFTER=<n>` (the page destroys its device after the n-th batch, to test recovery).

### Batch size: `B=auto`

`FrontHalf.create`'s `B` is `"auto"` by default, and so is the harness's (a job naming no `B` still gets 8 from
`scripts/gpu/harness/page.mjs`). WebGPU tells a page its buffer and workgroup limits, not the memory free, the clocks or how long
a submit may run before the device is reset, so the controller (`decoder.mjs` `Batcher`) measures them on the
decoder's own batches.

- Memory. The first batch prices a frame: a one-frame lane at that batch's frame and picture size, its back half
  lane's bytes shared out a frame (the back half is built at create for `B_CEIL`, so the lane holds 32 frames of it).
  Then two lanes of Bmax frames are made, the largest batch whose lanes fit the budget beside what is shared
  (`budget`, the device's own `maxBufferSize`, `budgetFor`: 2 GiB on the S26 Ultra, 4 on the 4090, 256 MiB on
  SwiftShader, WebGPU stating no memory size), never more than `B_CEIL` = 32 a batch, and capped by the device's
  buffer size and texture layer limits; an allocation that fails its out-of-memory scope is tried again at half,
  down to 1 (`makeLanes`). The back half's shaders are compiled for a batch size: at create for `B_CEIL`, and again
  in planning only for lanes memory holds under it (SwiftShader). A plan reads only the frames' shapes.
- Time. The stream starts at Bmax and the watch alone sizes it. The cap, `capMs` (400), is a fifth of the submit of
  about 2 s that lost the iGPU's device, which leaves room for throttling to double a batch's time, and for a batch
  dearer than the ones before it, before the watch shrinks it.
- Watch. Every batch but a lane's first (every pipeline's and buffer's first use): one over the cap halves the size at
  once, three running over 90% of the cap halve it, sixteen running with room for a double grow it back toward Bmax.
  A batch with a larger frame or picture than the lanes were planned for plans them again, once every lane is idle.
- In flight: 2 lanes.
- The ring (`decoder.mjs` `enqueue`): Bmax + 4 layers at the lanes' shape, counted in the budget
  beside the lanes, so Bmax is a little smaller with it where memory binds (SwiftShader; on the S26, the iGPU and the
  4090 `B_CEIL` binds first). A frame goes onto it the moment it arrives and the caller's bytes are free; a batch is
  cut from staged slots, its encoder beginning with a copy a slot into the lane's layer, and the slots are freed at
  the submit.
- Device lost. The harness's page makes a new device and a decoder that starts at half the batch that was running and never
  grows past it (`create` `restart`), and runs the lost batches again on it, in order.
- A batch carries any number of frames up to its lanes' B. Every dispatch covers only the frames it carries (the
  back half's soft values and LDPC too, `back.mjs` `encode` `frames`), clears and readbacks cover only them, and an
  empty slot exits at once; the back half's gate still walks every slot in one thread.
- The page asks node for jobs of the controller's size (`GET /job?want=n`, `scripts/gpu/harness/run.mjs` `autoJobs`) and cuts
  a larger one (the first, asked before the plan, or one asked before a change) into batches of the size wanted as
  each goes; its batches' results go back as one. The total line names the size, Bmax, the limit that bound it, the
  in-flight count and why, and the plan's time, which comes before the first submit and so outside the span.

### Batches in flight, and what the total line means

`INFLIGHT=<n>` (default auto: 2) is how many batches the page keeps on the device at once. `FrontHalf.create`'s
`inflight` makes that many lanes (a batch's working buffers, bind groups, timestamps and readback; pipelines,
tables and weights are compiled and uploaded once and shared), `run()` takes a free one and returns as soon as
the batch's frames are on the ring and the batch is submitted, its promise being the readback, and the page submits
the next batch while the ones before it run. A batch's readback is requested at its submit, so it never waits on a later submit. The
blocks are the same at any n; device memory is n times a lane (the wasm back half's lanes are the larger, most of
them the grid and its readback); a submit is still one batch, so n does not lengthen it (the iGPU loses its device on
a submit over about 2 s).

The total line is the throughput. Frames a second is the page's span, first submit to last readback, over the
real frames (empty slots count as nothing), each run's span summed; the steady rate beside it leaves out the
first batch (its frames and the wait for its readback), so it is the pipe once full. Compute ms a frame is the
stage timestamps, and the device busy share is the two multiplied: near 100% the device bounds the run, far below
it the host does. What the page waited for between batches says which host part: a job still on its way from node
(the HTTP trade: the frames out, and back the f32 grids with the wasm back half or the blocks and counters with the
GPU's; on the 4090 the trade, not the device, bounds a run) or a lane still busy (the device). `PREFETCH=<n>`
(default 2) is how many jobs the page keeps on their way from node at once; it moves only the transport-bound 4090.
A result's body is one buffer, not a Blob: a Blob's bytes stay in Chrome's blob store until its wrapper is
collected, and on a 600-frame run on the 4090 a post failed ("Failed to fetch"). The page also keeps at most
four results on their way back to node, and a wait for a post is the third wait on the total line: a long 4090
run is bound there at any n in flight, so its rate is the transport's, not the decoder's. Latency a batch is its upload to its readback, which n in flight lengthens; it is
not the number that matters. Under `PROFILE=1` a run also prints the page's side of a batch in ms: upload (the
frames into the texture), encode, scopes (the error scopes' round trip), map (the readback's wait), read (out
of the mapped range), a job's fetch, and the two waits.
