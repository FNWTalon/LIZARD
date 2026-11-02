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
verdict a block and the counters.

One symbol a frame; a 2:1 capture is two frames, a half each.

## Shape rules

- A batch is B frames as layers of one texture: one command encoder, one readback. Nothing in between waits on
  the host; later stages size themselves from counters earlier stages wrote.
- Every list has a fixed cap on the device. Hypotheses are enumerated and all scored; the frame's answer is an
  argmax. The finder runs twice at most: its first round on the proposer's levels 2 to 4, a second round with level 1
  on the frames the first found nothing on (`glsl/finder.mjs` REDO, which sizes every dispatch of the round: none when
  the first found every frame); past that, no early exit, no fallback chain, no retry, and a miss is counted.
- Float throughout. Discrete choices are argmaxes over float scores.
- Explicit bind group layouts, the host's; a GLSL local and workgroup memory hold nothing until written (a kernel
  zeroes workgroup memory only where it says so); a per-stage counter block comes back with every batch so a stage
  that did nothing reads as zeros.

## Stages (`glsl/`, their constants and plans in `plan/`)

| | file | what |
|---|---|---|
| F1 | pyramid | 2 x 2 means, levels 1 to 4 |
| F2 | bank | the peaks proposed to F4, each 16 x 16 tile of every level keeping its 4 strongest as (x, y, sigma, response). The trained fully convolutional proposer on levels 1 to 4 (`cnn/proposer/`), hosted by `bank_fcn2.mjs` to the contract in `glsl/bank.mjs`: each pixel standardised over its own 15 x 15 box, so the map is the level's alone and a workgroup takes a region of tiles; int8 where the device has the packed dot product |
| F3 | bank SELECT | the strongest `cap` peaks a frame, one cut over every level the round runs (the raw slots from level 1), the histogram then one append pass |
| F4 | classify | per peak: is it a corner mark, which way is out, how big a module. `classify` is the trained network (`cnn/`, cnn-v3), run on every kept peak. The kernel is `glsl/classify.mjs` over `plan/classify_gemm.mjs`'s plan: several patches a workgroup, a register tile a lane, a split conv reduced by subgroup shuffles where the device has them; `plan/classify.mjs` names the installed weights (`CLASSIFIER`) and their int8 twins |
| F5 | finder VOTE, PEAKS, GATHER | every reading with p at least `SMIN` votes along a ray toward the symbol's centre (the diagonals cross there under any perspective), one lane a reading; every crossing tallest within `SUPPRESS` cells goes into the frame's list, one lane a cell; the K strongest crossings each gather a quad in two rounds (the band about the crossing, then about the centre a filled diagonal pair gives); a quad is its four corners or nothing |
| F5d | finder COMPLETE | a quad of three corners where no quad of four was gathered: the fourth from the crossing of the ring's two sides read toward it (the corner a crop or the nets missed), held to SCORE as any quad |
| F5e | finder TRACE | each quad's bend: the ring's inner edge traced from each mark (a station every 12 modules, each predicted from those found and located by the gap, ring and rim's profile, in the marks' own module sizes), then one radial term about the camera's axis by the plumb line, in the map's own radial form; a lens without a strong bend traces close to 0 |
| F6 | finder SCORE, PICK | every (four-cornered quad, orientation, ring) read against the known timing track near the corners twice, through the marks' homography and through the quad's bend; argmax: the ring, never the picture; a bent winner's bend goes into the map |
| F7 | register NODES, REFIT, REFIT_POST | every border node correlated against the border as painted (256 lanes a node over offsets and sample groups, +-3 modules at half a module on one shared lattice about the finder's map); REFIT a homography by Gauss-Newton, every solve straight-line, then REFIT_POST the homography's first-order change and each side's residual fitted by a polynomial of 1 to 6 coefficients, the count chosen by AICc on the side's own nodes. One round (the anchors' stages search the border again about their own model, F7a); profiled as nodes and refit |
| F7a | anchors ANCH_FIND, ANCH_PREDICT, ANCH_REFIT, ANCH_FIT (every picture carries them, SPEC 6.13) | the 7 x 7 grid of anchor squares: after F7, each square found in a window one side of it wide each way about where the frame's final map puts it, by the pattern's normalised correlation (box sums of a window's integral images): coarse at every second point and step over the whole reach, then every point and step within 3 of the coarse best, a parabola each way; on a frame with 12 found or more (`ANCH_MIN`), ANCH_PREDICT fits the lens over the anchors found and predicts each border node from it, and the border is searched again about each prediction, +-0.5 module at a quarter (F7's node search, centred, on its lattice; a frame without the anchors keeps F7's nodes); ANCH_REFIT fits the lens about the camera's axis again over the border's nodes as measured and the anchors (K radial terms by AICc with no cap, the anchors sampling the interior's radii; robust over the border: a node further from the model than half a module or three robust spreads is dropped and reads the model, four fits at most), each side's curve of what it leaves on the border, and each anchor's residual against that map (`../src/acquire.c` `border_fill_anchors` is the same fit); then ANCH_FIT takes a frame's anchors' residuals whole and fills the missing ones from their neighbours, with the picture's mean, into the frame's record (`result` word 31 the anchors found). The back half's pass 1 then takes a cubic mesh of those residuals into every sample's position and sets the squares to the mean before the detrend |
| F8 | word | the format word, one code a ring over every word cell (`src/fmt.h`: RS(16,3), (32,3), (64,3) in the 64, 128 and 256 rings), each cell read at its planned centre against its track pairs' grey, ordered statistics decoding at order 3 over versions 2 to 128 and every rate, taken by a side-aware gate at the C's false-word rate (`wordcode.mjs` `gateTable`); the frame decodes at the word's version, or the batch's held one (the last word the host read) when none is taken, or not at all: `sel` = (ring, 1 + version, picture slot, 0), the block count and picture for the back half (after F7) |
| B1 | back gate (`glsl/back_transform.mjs`) | lists each built picture's frames by the slot F8 gave them (a frame not valid, with no version, or naming an unbuilt picture is on no list) and writes the indirect dispatch sizes (pass 1, reduce, pass 2 and the soft values a picture, the LDPC a code over every listed frame) and each frame's (w, h), blocks, rows and ring for the fused pass 1 (whose PIC holds each ring's grid for the picture and lattice); zeroes the counters, verdicts and record count, so the host clears nothing |
| B2 | back pass1, reduce, pass2 | pass 1 samples a column pair through the map and the Coons patch of the sides' residuals (`glsl/sample.mjs` COONS), made whole on a node grid every 8 rows at a quad's outer columns and linear between; the picture's 2D DFT kept as S = T / n^2 on the disc only: along y a column pair at a time (the column means apart, which keeps f16 safe), the detrend's five sums reduced, then along x a kept row at a time with the detrend subtracted in the spectrum; Y and S in f16 where the device has shader-f16 |
| B3 | back soft (`glsl/back_tiers.mjs` talign, tsoft, tblocks: the rate profile's stage, `back/tiers.mjs`) | the grid's shift fitted from the pilots in the blocks' tails and turned back; a sub-channel's 320 coefficients to int8 LLRs in slot order through their moments (the C's arithmetic, `src/lizard_phy.c`), whitened by slot, with its pilot reading; a block's estimate over its own sub-channels (7, 8, 9 or 12 by its code), the block declined under its code's bar (0.2 k / n). Every version's profile is a row of the stage's table, so any version at any picture decodes |
| B4 | back ldpc (`glsl/back_tiers.mjs` tldpc, a dispatch a code: 7/8, 3/4, 2/3, 1/2) | a codeword a workgroup: the bit map as one gather (L whitened by the soft stage), layered min-sum in i16 with the C's norm and caps (`src/ldpc.c`), the bits past the payload and CRC known zeros, each code's own learned stop (a small net trained on that code's neutral codewords, `back/stop/`, `back/DESIGN.md` 6), the CRC-32, and a verified block's 473 bytes compacted into the result record; one verdict and an iteration count a block |

`decoder.mjs` is the setup's host: `FrontHalf.create` makes every pipeline, bind group layout and static buffer a
device needs, on the recording device of `../gen/mockgpu.mjs`, and `../gen/gen.mjs` writes them out as the native
tree (`../out/`); `tables.mjs` takes the format's tables from the wasm codec at start: a table a ring (the border:
lattice, kind map, marks, track and word plans) and the eight pictures, apart. The native host (`../core/dec/`)
builds that tree on the device and runs the lanes and batches.

## Running

The decoder runs natively through `liblizard/core` (the Android and desktop apps' receivers, and `lizard_gpu_check`):
the JS host here builds its setup on a recording device (`liblizard/gen/gen.mjs`, `lizard-android/build.sh gen`),
every kernel's GLSL compiled to SPIR-V by glslang 16.6.0. The web receiver decodes with the C (`../src/`); the web
sender keeps the WebGPU encoder (`encoder.mjs`, its WGSL in `wgsl/`).

`bank_fcn2.mjs` takes the first proposer form the device's workgroup memory holds: f16 with a 32 x 32 region at 64
KB, f16 48 x 16 (31,340 B) at 32 KB, f32 16 x 16 at 32 KB, f32 arithmetic with its activations stored as halves on a
device without `shader-f16` that holds no f32 form, and `FitError` below that: no GPU decoder there, the C decodes.
The generated tree is made for 32 KB of workgroup memory, which every phone we know offers.

### Batch size

The host's batch controller (`../core/dec/front.cpp`) measures the device on the decoder's own batches.

- Memory. The first batch prices a frame: a one-frame lane at that batch's frame and picture size, its back half
  lane's bytes shared out a frame (the back half is built for `B_CEIL`, so the lane holds 32 frames of it). Then two
  lanes of Bmax frames are made, the largest batch whose lanes fit the budget beside what is shared, never more than
  `B_CEIL` = 32 a batch, and capped by the device's buffer size and texture layer limits; an allocation that fails is
  tried again at half, down to 1. The tree carries the back half built for 32, 16, 8, 4, 2 and 1 frames, and Bmax is
  rounded down to one of them.
- Time. The stream starts at Bmax and the watch alone sizes it. The cap, `CAP_MS` (400), is a fifth of the submit of
  about 2 s that lost the iGPU's device, which leaves room for throttling to double a batch's time, and for a batch
  dearer than the ones before it, before the watch shrinks it.
- Watch. Every batch but a lane's first (every pipeline's and buffer's first use): one over the cap halves the size at
  once, three running over 90% of the cap halve it, sixteen running with room for a double grow it back toward Bmax.
- In flight: 2 lanes.
- The ring: Bmax + 4 layers at the lanes' shape. A frame goes onto it the moment it arrives and the caller's bytes
  are free; a batch is cut from staged slots, its commands beginning with a copy a slot into the lane's layer, and the
  slots are freed at the submit.
- A batch carries any number of frames up to its lanes' B. Every dispatch covers only the frames it carries (the
  back half's after the gate only the frames the gate lists, by its ARGS), and an empty slot exits at once.
