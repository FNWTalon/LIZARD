# The GPU back half: design (2026-09-24)

Input: the n x n sampled picture the front half writes (F9, `wgsl/sample.mjs`), one a frame, B frames a batch; since
the fused pass 1 (section 12), the frame itself through the front half's map.
Output: CRC-verified blocks with their ids, and nothing else, read back once a batch (since 2026-09-30 also each
block's pilot reading, since 2026-10-01 each frame's grid shift). The reference for every rule
is `src/focus.c` `focus_finish_ext`, `src/ldpc.c` `ldpc_decode_stall`, `src/layout.c` `ob_crc32`.
The decoder is judged on blocks that pass their CRC-32 against the truth stream, on ms a frame at batch B (the
iGPU's; the 4090 counts blocks), and on a negative control; never on parity with the C. Every table named below
was verified today by calling the export from node (`init()` from `sim/ob.mjs`); the checks and their counts are in
sections 8 and 9.

Formats in play when this was written (2026-09-24): n / subch = 256/32, 512/128, 1024/560, 2048/1024 (the top of
each picture size then) and the recorded runs' 256/16, 512/64, 1024/192, 1024/256, 1024/512; the tables, sizes and
measurements of sections 1 to 12 are those formats' at one code rate, on recordings of the border before 2026-09-27
unless they are v0.3's. Since 2026-09-27 the pictures are 256, 384, 512, 768, 1024 and 1536 (`sim/lizard_pick.mjs`
`PICTURE_SIZES`; tops 32, 80, 136, 320, 568 and 1024 sub-channels), and 2048 is none. Every format then was ONE LDPC
code (rate 3/4: n 5088, k 3816, m 1272, z 106, 12 block rows of 15 data slots) and one block size, 473 payload
bytes, the first 4 the block id (LE), subch / 8 blocks a frame. Since 2026-10-07 the code rate follows the frequency
(`src/focus.h` `focus_tiers_for`): from the lowest frequencies out, 7/8 on 7 sub-channels a block (n 4464, z 93),
3/4 on 8, 2/3 on 9 (n 5760, z 120; since 2026-10-08) and 1/2 on 12 (n 7680, z 160), every block 473 bytes. Pass
one's soft values and LDPC are the rule stage since (`back/tiers.mjs`, `wgsl/back_tiers.mjs`): a frame's profile is
a row of its table by the frame's version, and the arithmetic of sections 1 and 4 to 7 runs a code at a time, each
code with its own block rows and its own learned stop. The one-rate soft and LDPC kernels below are deleted
(2026-10-22): `soft.mjs` and `ldpc.mjs` make only the tables and buffers the rule stage binds. So are the grid pass 1
(section 12) and the cancel stage (13); the standalone tests of sections 9 and 10 are in `archive/gpu-back-tests/`.

## 1. Stages, in order, and the arithmetic each reproduces

The picture z(x, y) is luminance in 0..1 (the C's gamma table at gamma 1 is i / 255; the sampler reads r8unorm).
Everything after the picture is scale free except the int8 quantum of the soft values, so a constant factor on z is
invisible. The spectrum is carried as S = T / n^2 (T the C's unscaled transform, whose values reach 5.7e5 on the
recorded frames, past f16's 65504); every constant of the detrend is divided the same way at the store.

- **A0 reduce**, per frame. cx[i] = (2 i + 1 - n) / n, qx[i] = cx[i]^2 - mean(cx^2), sx2 = sum cx^2, sq2 = sum qx^2
  (all from `focus_tables`). Five sums over the picture: Sx = sum z cx[x], Sy = sum z cx[y], Sxy = sum z cx[x] cx[y],
  Sxx = sum z qx[x], Syy = sum z qx[y]. Normalised: px = Sx / (n sx2), py = Sy / (n sx2), pxy = Sxy / sx2^2,
  pxx = Sxx / (n sq2), pyy = Syy / (n sq2). The C sums a row in four f32 chains and carries on in double; here f32
  tree reductions. Drift moves the detrend by parts in 1e-6: fine. A1 produces the partials at its load, so the
  picture is read once.
- **A1 transform along y**, per picture column x: Z_x(v) = (1 / n) sum_y z(x, y) e^(-2 pi i v y / n), kept for
  v < V only (V = vmax + 1, section 3). Two real columns share one complex FFT, a + i b, separated by conjugate
  symmetry: A(v) = (Z(v) + conj Z(n - v)) / 2, B(v) = -i (Z(v) - conj Z(n - v)) / 2 (`unpack_turn`). Each column's
  mean is subtracted before the FFT and stored on its own as Z_x(0) = mean (exactly that value), which is what
  makes f16 safe for the rest (section 3).
- **A2 transform along x**, per kept row v: S(u, v) = (1 / n) sum_x Z_x(v) e^(-2 pi i u x / n), stored for the disc
  only (section 3). The detrend goes in the epilogue, in f32, in the C's units T = n^2 S before the store: for every
  (u, v): T -= pxy X1[u] X1[v]; on v = 0: T(u, 0) -= n (px X1[u] + pxx X2[u]); on u = 0: T(0, v) -= n (py X1[v] +
  pyy X2[v]). X1, X2 are the DFTs of cx and qx (`focus_tables`, complex, index (u + n) mod n). T(0, 0) carries
  nothing and is not stored. This is the C's spectral subtraction (focus.c `focus_finish_ext`) to the letter; a
  spatial subtraction of the same surface before A1 is identical by linearity, and it is the option only if a
  pass over the picture ahead of A1 ever becomes free (it costs the picture's bytes again, and it breaks the
  fused sampler, which has no picture to pre-pass).
- **B soft values**, per block b (first sub-channel s0 = 8 b), per sub-channel j = s0 + g over its 320
  coefficients y_s = S at pos[320 j + s]: m2 = mean |y|^2, m4 = mean |y|^4, a2 = sqrt(max(2 m2^2 - m4, 0)),
  nv = max(m2 - a2, 0.02 m2), k = 0.7 * 2 sqrt(2 a2) / nv, LLR = clamp(k y, -10, 10) per axis, int8 = round half to
  even (WGSL `round`, the C's `nearbyintf`) of 8 LLR; out[2 s] the real axis, out[2 s + 1] the imaginary: slot order,
  the C's `ext_llr`. est_j = J(a2 / nv), J(g) = (1 - 2^(-0.3073 (2 sqrt g)^1.7870))^1.1064 for g > 0, else 0.
  blk_est = mean of the block's 8 est. f32 throughout (the C: double). 2 m2^2 - m4 is a difference of near equals
  only where nv is at its floor, a block that declines anyway.
- **B decline gate**: declined = blk_est < 0.2 k / n = 0.15 (`FOCUS_DECLINE`). A block within 1e-6 of the bar may
  differ from the C. Accepted.
- **C bit map and whitening** (`map_in`): codeword bit j of block b is slot inv[j] of the block, sign flipped where
  white[5120 b + inv[j]] = 1. perm[i] = i P mod 5088 with P = 1943 (`perm_fill`, LINEAR), inv = perm^-1; white =
  PRBS-23, x^23 + x^18 + 1 from the all-ones state, one bit a slot of the frame in pos order (`whiten_fill`). Bit
  map 0 (every recording before 2026-09-23, `recordedSpec`): inv is the identity and nothing is flipped. Slots 5088
  to 5119 of a block carried whitened zeros and were not read; since 2026-09-30 they carry the pilots (the
  whitening XOR a bit of the painted picture's count, SPEC 7.3), which the soft stage reads and, since 2026-10-01,
  fits the grid's shift from (16 such slots at 7/8, none at 2/3 and 1/2, whose codewords fill their blocks).
  k = 3816 = 473 * 8 + 32 exactly, so the C's "known zeros past the payload" loop (focus.c `focus_finish_ext`)
  is empty at this rate; the 7/8, 2/3 and 1/2 codes' k (3906, 3840, 3840) is roomier, and their bits past 3816 go in
  as known zeros at 127, as the C's do.
- **C LDPC**: normalised min-sum, layered by block row, norm 13/16, posteriors clamped to +-8191, messages to
  +-127, cap 30 iterations, sweep direction alternating each iteration, the stall rule (violated checks after
  iteration 9 above 0.95 of the count after iteration 1 gives up, `FOCUS_STALL_IT`, `FOCUS_STALL_RATIO`), exit on a
  clear syndrome. Integer arithmetic, kept integer: the norm and caps were set on 78 000 blocks
  (`test/ldpc_ablate.c`, focus.c `focus_finish_ext`) and mean what they were measured to mean only in integers.
  Given equal LLRs the decisions and the iteration count equal the C's bit for bit under the stop rule `"c"`; the
  first read's default since 2026-09-28 is the learned rule (section 6; the funnel from 2026-09-24; each code of the
  rate profile its own net since 2026-10-07), which gives up on a failing codeword by a small net's reading and
  verifies the same codewords but for its budget of 1 in 1,000.
- **C CRC-32 and compaction**: the 473 payload bytes are systematic bits 0..3783, MSB first in each byte; the CRC
  is bits 3784..3815 MSB first; `ob_crc32` is the reflected polynomial 0xEDB88320, init and final xor 0xFFFFFFFF.
  Equal means verified; a verified block claims a record slot and is written; nothing else leaves the device.

Drift is acceptable everywhere before the int8 quantisation and in the quantisation itself: a slot within f32
noise of a rounding boundary may differ by one, and a marginal block may then decode or not. The acceptance
criterion is blocks against the truth stream on the same frames (section 10), not equal bits. Not in this
design: the decision-directed pilot pass (`dd_mode`, a second LDPC pass through a shift the verified blocks
imply; built 2026-09-27 as the shift stage and deleted 2026-09-28, section 14) and burst accumulation. The GPU reads
every frame once.

## 2. Buffers between stages

Frame f of a batch of B (frames may differ in picture size; every stage reads `sel[f]` and `frames[f].valid` as
the sampler does). `sel[f]` is a vec4u since 2026-09-27 (the three rings): (ring, 1 + version, picture slot, 0). PICK
(F6) writes (ring, found, 0, 0); F8 (`gpu/wgsl/word.mjs`) then (ring, 1 + v, slot, 0) for the version v its word or
the batch's held word names, else y 0. A frame is sampled and listed when y >= 2; z indexes the picture sizes. Per frame at the top format of each n; a batch is B of each, plus the shared tables.

| name | writer -> reader | element and order | 256/32 | 512/128 | 1024/560 | 2048/1024 |
|---|---|---|---|---|---|---|
| P | F9 -> A1 (not made with the fused pass 1, section 12) | f32 at f * gridStride, strip order: (x div 64) n 64 + y 64 + x mod 64, 0..1 | 256 KB | 1 MB | 4 MB | 16 MB |
| LISTS, ARGS | gate -> all | u32 [SLOTS sizes][1 + B] (SLOTS = 6 since 2026-09-27, `wgsl/common.mjs`): count, then frame indices, then each frame's (w, h) and each frame's block count (2026-09-26: the version F8 read in sel.y = 1 + v (or the held one, 2026-09-27), up to the size's blocks, which are 8-sub-channel chunks; a frame with neither is not listed since 2026-09-27, the size's count before; soft and LDPC ran a frame's first count blocks, and the rule stage, since 2026-10-07, the profile of that version), then each frame's row count (2026-09-26, `listsRows`: the disc rows that count takes, section 3 "Rows a frame"); u32 [SLOTS][12] indirect args, (x, y, z) for pass 1, reduce, pass 2 (the most rows a listed frame of the size takes) and the blocks slot (blocks, 1, count) | 376 B + 288 B at B = 8 (`listsWords`, `argsWords`) | | | |
| PART | A1 -> reduce, A2 | f32 [n][4] a column: (sum z, sum z cx[y], sum z qx[y], the column mean); the mean is what was Y0 (folded into w on 2026-09-25, section 13: the binding it held goes to the cancel variant's REF) | 4 KB | 8 KB | 16 KB | 32 KB |
| COEF | reduce -> A2 | f32 [8]: px, py, pxy, pxx, pyy, 0, 0, 0 | 32 B | | | |
| Y | A1 -> A2 | complex, v major: (v, x) at (v - 1) n + x, 1 <= v < V; f16x2 (f32x2 on the 4090); V rows of room a frame (2026-09-25: the cancel stage's inverse rows fill a frame's slot with a whole disc, rows 0 to V - 1, section 13) | 82 KB | 330 KB | 1.38 MB | 3.74 MB |
| S | A2 -> B | the disc, one entry a used coefficient, row compact (section 3); f16x2 or f32x2 | 41 KB | 164 KB | 717 KB | 1.31 MB |
| L | B -> C | int8 [subch][640] packed 4 a u32, slot order (the C's `ext_llr`) | 20 KB | 82 KB | 358 KB | 655 KB |
| EST, BLK | B -> C, host | f32 [subch]; a block: (f32 est, u32 declined) | 0.2 KB | 0.6 KB | 2.8 KB | 5 KB |
| V, ITS | C -> host | one 32-bit word a block each, at f * blocksMax + b: u32 verdict (0 not run, 1 verified, 2 declined, 3 stall or cap, 4 CRC), i32 iterations (the C's codes) | 16 B each | 64 B | 280 B | 512 B |
| REC | C -> host | u32 count, then 484 B records: word 0 = frame + 65536 block, word 1 = iterations, 473 payload bytes (id first), 3 pad | cap = blocks in the batch x 484 B | | | |
| BCOUNTS | all -> host | u32 [B][8]: columns transformed, rows transformed (the frame's rows since 2026-09-26), coefficients stored, blocks tried, verified, records, sub-channels quantised, iterations run | 32 B | | | |

S holds subch * 320 entries exactly (10240, 40960, 179200, 327680 at the four tops; verified against `npos`), 31 to
51% of the C's half plane (vblocks * n * 64). Per batch of 8, f16: at 1024/560 Y 11 MB, S 5.7 MB, L 2.9 MB, REC cap
271 KB; at 2048/1024 Y 30 MB, S 10.5 MB, L 5.2 MB, REC cap 0.5 MB; at 512/128 Y 2.6 MB, S 1.3 MB, L 0.7 MB. P exists
already (the front half's `gridBuf`, sized for nmax). V and ITS were planned as a byte a block; WGSL has no byte
store, so each is a word (section C as built), and a frame's rows sit at the largest built size's blocksMax. The
host was to read count * 484 bytes of REC; as composed it copies the whole of REC in the batch's one readback,
because the count is known only on the device (section 11 has what that costs a frame).

## 3. The transform

Coefficients needed: the half disc u^2 + v^2 <= R^2, v >= 0, without v = 0 with u <= 0, R = sqrt(640 subch / pi),
taken in order of r^2, then v, then u, the first subch * 320 (`init` in focus.c). Enumerated from `focus_pos_ptr`:

| n / subch | R | vmax = umax | V rows | coefficients | disc / (n V) |
|---|---|---|---|---|---|
| 256 / 32 | 80.7 | 80 | 81 | 10240 | 49% |
| 512 / 128 | 161.5 | 161 | 162 | 40960 | 49% |
| 1024 / 560 | 337.8 | 337 | 338 | 179200 | 52% |
| 2048 / 1024 | 456.7 | 456 | 457 | 327680 | 35% |
| 256/16, 512/64, 1024/192, 1024/256, 1024/512 | 57.1, 114.2, 197.8, 228.4, 323.0 | 57, 114, 197, 228, 322 | 58, 115, 198, 229, 323 | 320 subch | 34, 35, 30, 35, 50% |

S layout: row v at off[v], v = 0 holds u = 1..umax(0), v > 0 holds u = 0..umax(v) then u = -umax(v)..-1, with
umax(v) = floor(sqrt(R^2 - v^2)) capped so the count is exactly subch * 320 (`discLayout` in `transform.mjs`
builds off, umax and the coefficient-to-entry table from pos; every entry of pos maps to exactly one entry of S).

Order: along y first, then x, as the C does. Both orders cost the same FFTs: n / 2 real pairs of length n, then V
rows of length n (x first keeps u = 0..umax of each real row and recovers u < 0 from T(-u, v) = conj T(u, -v)).
What differs is traffic against the strip order, where a row is 256 B runs and a column a 4 B stride: at
1024/560, f16, y first moves about 10.8 MB a frame (the picture read at 2x sector amplification in 4-column
tiles) and x first 8.2 MB, but once the sampler is fused into pass 1 the picture is not read at all and y first
moves 2.8 MB to x first's 4.2 MB (its pass 2 then reads Y at a stride). y first is the target shape; if pass 1's
read stands out in the profile before the fusion, widen the column tile (8 columns fit f16 workgroup memory at
n <= 1024) before considering the other order. Output pruning inside a pass is worth at most the last stage
(the band holds every residue until 2^s exceeds 2 umax + 1) and is not in the plan; the pruning is which rows
exist (V of n / 2) and what is stored (the disc).

Kernel shape: one column pair in workgroup memory, Stockham, eight points a thread in registers, one exchange
through workgroup memory a stage, no ping-pong buffer. Threads a pair = n / 8 (32 to 256). Stages: 256 = 8 8 4,
512 = 8 8 8, 1024 = 8 8 8 2, 2048 = 8 8 8 4 (radix 16 for the last two where registers allow; `archive/stash/fft-gen2`
and VkFFT are the references for the butterflies and the twiddle handling). Since 2026-09-27 also 384 = 8 8 2 3,
768 = 8 8 4 3 and 1536 = 8 8 8 3, the radix-3 stage last and in place (`radix3Stage`). Workgroup memory a pair: n * 8 B in
f32, n * 4 B in f16. A workgroup takes 4 adjacent picture columns (2 pairs, in turn in the same memory) at
n <= 1024 and 2 at 2048; the sampler fusion removes the sector waste entirely. Pass 1 writes Y with x contiguous
in runs of 4 (the L2 merges neighbouring workgroups' partial lines). Pass 2 reads a row of Y contiguous (the
v = 0 row from Y0 as f32) and writes its compact disc row. Twiddles: a quarter-wave table (n / 4 complex f32) in
a storage buffer read once a butterfly; the 2026-09-22 port found a table slower, but that transform was
storage bound; measure both. A pipeline a pass for each n (four then, six since 2026-09-27), dispatched indirectly
from ARGS over LISTS; a workgroup for a frame of another size returns before its first barrier.

Rows a frame (2026-09-26). Every size is built at its top (`gpu/tables.mjs` CURRENT) and a frame decodes the
version its word names, a prefix of the top's coefficients, so it needs only the rows its blocks' coefficients
reach. `transform.mjs rowsByCount` gives rowsBy[c], the rows the first c blocks take (block b is pos[2560 b ..
2560 b + 2559], sub-channels 8b to 8b + 7: an 8-sub-channel chunk, the version's unit, whatever the rate profile's
blocks span), from the top's own pos table; `scripts/gpu/back/check_rows.mjs` holds every version
of every size to its own disc's V (128 of 128 over the six sizes, 2026-09-27). The
gate's uniform carries it as rows[c][size]; the gate writes each frame's rows beside its block count in LISTS
(`listsRows`), pass 1 stores Y rows 1 to rows - 1 (the FFT itself is unchanged: every column still runs whole), pass 2
returns on a row past the frame's before its first barrier (LISTS is read-only there, so the branch is uniform) and
its indirect dispatch takes the most rows any listed frame of the size needs. S past a frame's rows is left stale:
the soft stage reads only its first count blocks' coefficients, all inside them. gate2 copies the rows into LISTS2
and sizes pass 2b the same way. The rule since 2026-09-26 (every frame decodes on its own, no memory across
frames or batches) makes the word the only source: a frame whose word is not read decodes at the batch's held word,
or, with none held, is not listed (since 2026-09-27; until then it ran its size's top rows and blocks, which kept
the recordings older than the word, every timing run of the protocol, at the top's cost).
irows and ipic (the sender's inverse transforms, `wgsl/inverse.mjs`) run the size's V: a symbol's rows past its version are zero.

Precision: f16 for Y, S and the workgroup-memory exchange where `shader-f16` exists (the iGPU, phones), f32 on
the 4090; f32 for the butterfly arithmetic, the partials, the reduce and the detrend epilogue. Why the means
come out first: the picture's mean is about 0.5 against a data rms near 0.125 and a single coefficient near
3e-4 of n^2; carried through f16 stages, the mean's rounding (relative 2^-12 of the sub-DFT DC at every stage,
landing on bins 0 and 2^s) sits within about 25 dB of the data on the v = 0 row and about 37 dB at v = n / 4.
With each column's mean taken out ahead of pass 1 and the v = 0 row kept in f32 through both passes, the f16
floor is about 50 dB under the strongest ring, and blur takes the outer rings 20 to 30 dB down from there. A
budget, not a result: the test in section 9 measures the error against the C's f32 spectrum on real frames
(target under -40 dB rms over the disc; f32 lands near -130, the 2026-09-22 port measured -137).

What the fused sampler (built: section 12) needs from the sampler's side: (1) its per frame inputs bound as F9 binds them
today (the texture array, `frames`, `maps`, `sizes`, `sel`, `resid`, `lattice`, `LT`); (2) `toImage` (common.mjs
MAP) and `coons` (sample.mjs) as importable WGSL strings behind one function `samplePicture(f, size, x, y) -> f32`
reading module (g0 + x step, g0 + y step) through `bilin0`; (3) the size selection per frame as it is (`sel[f]`).
Pass 1 then samples its 4 adjacent columns for all y and writes Y directly: no picture buffer, no strip order,
no readback of the grid, and the partials come from pass 1's load as they do now. Nothing in A2, B or C changes.

## 4. Soft values

Formula and moments: section 1. One workgroup a sub-channel, 64 lanes at 5 coefficients each: gather 320
coefficients from S through the UV table (u32 an entry: the S index of pos[i], from `discLayout`; subch * 320
entries, 1.3 MB at 1024 sub-channels, read once a coefficient a frame), tree reduce m2 and m4, one lane computes
a2, nv, k, est, every lane quantises its coefficients and writes two int8 (four slots a u32 store: a lane's 5
coefficients are 10 consecutive slots, so the pack is lane local). est to EST; the last workgroup of a block
(an atomic count of 8) writes BLK (est mean, declined).

The bit map and whitening as one gather table: MAP[b][j] = inv[j] | (white[5120 b + inv[j]] << 15), u16, one row
a block (5088 entries, 10 KB; 128 rows at LIZARD-1024, 1.3 MB, read once a codeword a frame, 3% of one LDPC
iteration's workgroup traffic). C's LDPC loads L[j] = (bit 15 ? -1 : 1) * llr[640 s0 + (MAP & 0x7fff)] and never
sees a slot again. Bit map 0: MAP[b][j] = j. L stays in the C's slot order so B is tested against the C directly
(`focus_rx_ldpc`), and C was tested on the C's own decisions (`focus_rx_bits`, an export that went with those tests
on 2026-10-22).

## 5. The decline gate

BLK.declined a block from blk_est < 0.15. C's workgroup for a declined block writes verdict 2, ITS -2, and returns
before touching L: a workgroup with no work, which the shape rules allow. The gate and the stall rule were the two
thresholds this decoder kept from the C (the stall rule left the first read for the learned stop on 2026-09-28,
section 6). They are format decoder rules measured on 78 000 blocks, and on the
hard runs the iterations they save are the LDPC's largest cost: on 02-36-59 (60 frames, 1920 blocks) the C
stalls 1076 blocks at 9 iterations, runs 716 to the cap and verifies 128 at 1.1 mean; on 21-34-51 (3840 blocks)
758 stall, 798 cap, 2283 verify at 3.2 mean, 1 declines. That is 520 and 630 full iterations a frame.

## 6. LDPC

H is never stored. The device holds the block row tables from `focus_ldpc_tables`: lay_ptr[13], then 180
(column, shift) pairs; every one of the 12 block rows has 15 data slots and 2 staircase slots (verified). Check
(r, i) reads data slot e as codeword bit col_e * 106 + (i - shift_e + 106) mod 106, its own parity bit
k + 12 i + r, and the previous check's k + 12 i + r - 1; for r = 0 the previous is (11, i - 1), and at i = 0 it is
absent (the C models it as a certain zero, posterior 8191, message re-zeroed every layer, and never scatters it
back; skipping the slot is the same arithmetic). A JS encoder built from exactly these tables reproduces what
the C paints on every slot (section 9), so the H rebuilt from these tables is the C's.

One workgroup a codeword, 128 threads, lane i < 106 is check position i. Workgroup memory: L as i32 [5088],
20352 B, data in code order, parity transposed as the C keeps it, Lp[r * 106 + i] = bit k + 12 i + r, so a slot's
lanes touch consecutive words; R compressed as 2 u32 a check [12][106], 10176 B: word 0 = min1 (bits 0..7),
min2 (8..15), arg (16..20); word 1 = the 17 message signs. R_e is rebuilt as (e == arg ? min2 : min1) with sign
bit e. Total 30528 B: under the iGPU's 65536 and the 4090's 49152 (both from the harness's adapter line), and
under the 32768 Adreno, Mali and Apple report in Chrome, to be confirmed by the same line on the S26. Below
32 KB: L as two i16 a u32 (10176 B, total 20352) at the cost of unpacking. Where f16 exists L can be f16 (10 KB):
posteriors above 2048 lose low bits, drift under the message cap; a variant to measure, not the baseline.

Schedule, per iteration: rows r = 0..11 (odd iterations 11..0). Per row, lane i: for each of the 17 slots
q_e = L[idx_e] - R_e; track min1, min2, arg, sign parity; m1 = min(127, (min1 * 13) >> 4), m2 likewise;
R_e = (parity xor sign(q_e)) ? -m : m; L[idx_e] = clamp(q_e + R_e, -8191, 8191); write the compressed record.
Within a row no two lanes touch one bit (a circulant is a permutation; the staircase bits belong to one check
each), so one barrier a row: 12 an iteration. Syndrome: the exact check is 12 rows re-gathered as signs only
(one third of an iteration's reads, no writes) XORed into a workgroup atomic; it runs every iteration in the
baseline. Cheaper, to measure: each row's parity from the updated signs at its own update into a flag, and the
exact check only on an iteration whose flags were all clear (the flags are neither necessary nor sufficient, so
this can delay an exit by an iteration and never admits a wrong codeword). After iterations 1 and 9 the exact
pass also counts violated checks for the stall rule. The exit is `if (workgroupUniformLoad(&done)) { break; }` at
the top of the loop (the atomic driven break the analyser refused in the 2026-09-22 port). Cap 30. Initial
L = the int8 LLR through MAP, R = 0.

Blocks a frame: subch / 8 (2 to 128). Codewords a batch: the batch's blocks summed over its frames, all sizes in
one dispatch over LISTS (workgroup (x, y, s) is block x of frame LISTS[s][1 + y]); LIZARD-256 at B = 8 is 256
workgroups, where the earlier port's 4090 sweep still scaled linearly. Cost model: a row is 17 slots * 106
lanes; an iteration 12 rows, 21.6 K lane slots; a failing block 30 iterations, 650 K; the recorded hard runs
above are 11 to 14 M lane slots a frame. Variant to measure on the iGPU if occupancy is the limit: two codewords
a workgroup of 256 threads (a finished codeword idles its half).

### The stop rule (learned since 2026-09-28; the funnel before it)

`build(..., { stop })` in `ldpc.mjs` picks when a codeword that has not cleared is given up, and a page's
`?ldpcstop=` overrides it (`BackHalf.build` passes none, so the decoder takes the default). A give-up is a stall to
everything after it (verdict 3, ITS -3). A rule can only give up: a block counts only when its syndrome is zero and its
CRC passes, so no rule makes a wrong block verify, and what a rule costs is the blocks it gives up that would clear.
- `"learned"`, the first read's default since 2026-09-28: after each iteration 1 to 29 (the cap ends 30) lane 0
  writes x = (bad, low, first, prev, t, est) to workgroup memory (the violated checks now, their running minimum,
  after iteration 1, after the iteration before, t, and the block's estimate `BLK.x`); lane j < 64 computes hidden unit
  j of the first layer, a barrier, of the second and its product with the output weight, a barrier, and lane 0 sums
  the 64 products in unit order and gives up when the sum is under T. No stall test. The net's 4,673 words (18.7 KB,
  `stop/rule.mjs` `netWords`, each layer input-major so the lanes read consecutive words) sit in MAP behind the bit
  maps, so the stage keeps its 8 storage buffers; 536 B more workgroup memory (i16: 21,384 B). The flags variants are
  refused with it (the rule reads the exact count after every iteration). The net, its trainer and its file are
  `stop/` (`scripts/gpu/back/stop/README.md`). Each code of the rate profile has its own net, trained the same way
  on that code's neutral channel to its plateau (`stop/stop_78.safetensors`, `stop_23.safetensors` and
  `stop_12.safetensors` beside the 3/4 code's `stop.safetensors`, picked by the code's checks, `stop/rule.mjs`
  `ruleUrlFor`; since 2026-10-07, the 2/3 code's since 2026-10-08), their words one after another in the rule stage's
  table (`back/tiers.mjs` `ruleContents`); the 1/2 code's kernel with its net needs 31,748 B of workgroup memory.
- `"c"`, the C's rule: the stall test after iteration 9 and the cap of 30. The check that the kernel's min-sum is the
  C's: under it `test_ldpc` held the one-rate kernel to `ref_ldpc.mjs`'s JS decoder and that to the C (`rxLdpc`) on
  every block (the test archived and the JS decoder deleted 2026-10-22, with that kernel); the rule stage takes it for
  every code (`?ldpcstop=c`).
- Pass two kept its own rule, `"funnel2"`, until it went with section 13 (2026-10-22).

The learned rule, as set for the 3/4 code. The question: given what lane 0 can hold after an iteration, is the
codeword hopeless? It is answered on neutral codewords only (`scripts/gpu/back/stop/neutral.py`: the format's code through a generic channel and the soft
stage's own demapper, channel information half near the decoding threshold and half from hopeless to clean, uneven
and dead sub-channels, fades the soft stage cannot see apart, gain and erasures; no recorded run informs a range):
524,288 codewords to train, as many to set the threshold (val) and as many held out (test), each traced through an
exact integer emulation of the decoder (`scripts/gpu/back/stop/intdec.py`, every dump record the C's replay's) with no stop to the cap.
Every rule that reads only the count and the estimate is scored exactly from those traces. The net: features bad,
low, first, fall = prev - bad (as fractions of m = 1272), t and est, standardised; two hidden ReLU layers of 64; one
logit; clearing codewords' states weighted 1,000 (the threshold sits among them); Adam at 3e-3 with a cosine schedule,
batches of 65,536, 64 epochs, seed 0. Trained to a plateau: the plateau sweep (three seeds a configuration) went
flat at 64 x 64 between 64 and 128 epochs (6.179 and
6.177 neutral val iterations, a seed spread of 0.02; 32 x 32 6.190 at 128; 128 x 128 no better), and seed 0 was the
best of its three on neutral val. The kernel's form folds the standardisation, 1 / m and fall into the first layer in
float64 (one rounding to float32) and the output bias into T; the threshold is set in that float32 arithmetic by the
criterion: the largest T whose lost clearing codewords stay at or under the budget times val's clearing codewords.
The budget is 1 in 1,000 (2026-09-28), chosen with the curve below in view. T = 0.19826725125312805.

Against the rules that learn nothing, on the same traces (iterations a codeword, then the blocks given up that clear
with no stop; the three dumps are the codewords the GPU ran on 20-04-09, 06-25-58 and 21-34-51, the old border, which
the net never saw and the funnel's counts were fitted on):

| set | none | the C's stall test | the funnel | learned |
|---|---|---|---|---|
| neutral val (sets T) | 21.689 | 13.950, 357 | 7.477, 1,709 | 6.151, 185 |
| neutral test | 21.659 | 13.904, 386 | 7.441, 1,815 | 6.130, 165 |
| 20-04-09 (1,920) | 9.046 | 5.907, 0 | 5.377, 0 | 5.730, 0 |
| 06-25-58 (2,619) | 15.574 | 8.221, 0 | 4.040, 0 | 3.367, 0 |
| 21-34-51 (3,648) | 4.358 | 3.858, 4 | 3.820, 4 | 3.900, 0 |
| the dumps pooled (8,187) | 9.045 | 5.734, 4 | 4.255, 4 | 4.159, 0 |

The budget's curve (T set on val each time): 1 in 10,000 val 18 lost at 8.048, the dumps 0 lost at 4.757; 1 in 1,000
185 at 6.151, 0 at 4.159; 1 in 100 1,859 at 4.267, the dumps 4 lost (21-34-51) at 3.443. The logistic rule the first
study fitted on the same features (`scripts/gpu/back/stop/README.md`) gave 7.510 and 4.341 at 1 in 1,000.

The arithmetic. Float32 in one order everywhere (the kernel, `stop/rule.mjs` `scoreF32` for the JS mirror,
`scripts/gpu/back/stop/net.py` `score` for the tables). Against float64 the decisions differ on 2 of 10,847,189 val states and 1 of
10,831,452 test states, and a compiler that fuses each multiply-add moves 1 and 0; none of the dumps' 65,867, whose
nearest state sits 3.3e-4 from T (the float32 error in the score reaches 3.4e-5). A 64-term sum cannot promise equal
decisions on every state; the flips sit within 5e-7 of T. `scripts/gpu/back/stop/check_rule.mjs` holds `rule.mjs` to `fit.py`'s words
and scores bit for bit (4,673 words; 6,166 states, the test set's 2,048 nearest T among them).

Checked on the GPU (2026-09-28, `research/results/ldpc_stop/gpu/`; bad 0 in every run):
- `scripts/gpu/back/test_ldpc.mjs` on the 4090 (the C's grids and soft values, B 8): 96 frames of each v0.3 recording, every block's
  verdict and iteration count the mirror's (8,064 of 8,064), 0 of the C's verified blocks given up, 2 more verified
  than the C's stall test (17-59-29), zeroed frames nothing, the noise curve equal at every level; under `STOP=c` the
  mirror equals the C and the GPU the mirror on every block (16 frames). On those 192 frames the funnel's tree
  (`archive/before-copies/ldpc_learned/funnel/`) ran 11,638 and 19,323 iterations (2.886 and 4.792 a codeword), the learned rule
  8,347 and 22,074 (2.070 and 5.475): 28% fewer on 07-56-38, 14% more on 17-59-29, where it verified 2 blocks more.
- The whole decoder on the 4090 (f32, B auto, every frame): 07-56-38 18,855 verified against the funnel's 18,844
  (`research/results/prune/after/`), 17-59-29 19,979 against 19,966, every frame found and worded both; `gpu_front 16`
  find 9,693 against 9,698 (one block less in each of 1080, k1=-0.045, bg noise g2, bg noise g8 and LIZARD-560 in
  128), truth 7,704 both, controls 0, words wrong 0.
- The iGPU (f16, 48 frames, B auto, `PROFILE=1`, the exclusive lock, the funnel's tree and this one interleaved
  twice): 07-56-38 compute 7.42 and 7.42 ms a frame to 7.11 and 7.13, LDPC 1.51 to 1.21; 17-59-29 8.58 and 8.57 to
  9.09 and 9.09, LDPC 2.68 to 3.19; blocks 1,159 and 1,573 in every run. Blocks per compute ms +4.2% on 07-56-38,
  -5.7% on 17-59-29.
- The net's own cost an iteration (the stage alone, `test_ldpc` `IGPU=1`, i16, B 8, batches with the same
  iterations under both rules): 13.72 to 14.08, 12.97 to 13.32 and 12.22 to 12.63 us an iteration at 654, 1,381 and
  about 4,240 iterations a batch (+2.6 to +3.4%), against the 12.4 of 2026-09-24.

### The funnel it replaced (2026-09-24 to 2026-09-28)

Seven checkpoints on top of the C's stall test: after iteration j, a codeword whose fewest violated checks so far is
still at or above f m is given up; (j, f) = (1, 0.35), (2, 0.32), (4, 0.315), (9, 0.29), (12, 0.27), (15, 0.26),
(20, 0.225), counts 446, 408, 401, 369, 344, 331 and 287 of m = 1272, each 26 to 33 checks over the highest running
minimum of any codeword that went on to verify on every frame of every recorded run (no frame held out). It gave up
4 verifiable blocks of the three dumps (21-34-51), where the learned rule gives up none. Its measurements:

Where the iterations went, from `scripts/exp/ldpc_stop.mjs` (archived 2026-10-22). The recorded frames went through the GPU decoder on the 4090
(FINDER=cnn, the defaults) with the grids read back. Each grid went through the wasm's soft values and a JS copy
of this decoder (`ref_ldpc.mjs makeTracer`, deleted 2026-10-22), run to the cap with no stall rule, which keeps each codeword's count
of violated checks after every iteration. The copy's verdicts equal the GPU's on every block but 6 in every frame
of every run (93,119 against 93,123 verified; the GPU's f32 soft values against the wasm's double, on the odd
marginal block), and its iteration totals equal the GPU's to 0.02%. A rule that reads only the count changes
nothing before it fires, so every such rule is scored exactly from these traces. Under the C's rule:

| | codewords tried | verified: share of codewords, share of iterations, mean iterations | stalled at 9 | run to the cap |
|---|---|---|---|---|
| 19 x 40 set (753 frames) | 19,698 | 58%, 18.5%, 2.9 | 4,876, 24.5% | 3,397, 57.0% |
| 02-36-59 and 21-34-51, every frame (1,200) | 57,566 | 52%, 17.7%, 3.6 | 14,985, 21.9% | 12,403, 60.4% |
| every frame of every run (5,942) | 170,639 | 55%, 20.4%, 3.8 | 45,919, 24.2% | 31,601, 55.4% |

No codeword cleared with a failing CRC. A codeword run to the cap mostly has its count wandering between 0.25 m
and 0.35 m for 30 iterations; it passed the stall test at 9 on the count's own noise. What separates the two
kinds is the running minimum of the count. Take the codewords that went on to verify and that the C's stall test
keeps. Over every frame of every run, the highest running minimum among them after 1, 2, 4, 9, 12, 15 and 20
iterations is 415, 375, 372, 343, 317, 300 and 256 checks. The failures' median at the same points is 415, 394,
380, 367, 363, 360 and 356. Each checkpoint sits 26 to 33 checks (at least 0.02 m) above that highest value.

The first constants were set on the 19 x 40 set and the two hard runs alone: 446, 395, 382, 344, 331, 318 and 268
at the same iterations, 25 or more checks above those sets' highest values. Every frame of every run then showed
margins of 31, 20, 10, 1, 14, 18 and 12. The 1 is 22-28-38 frame 293 block 0: its count goes 383 372 375 453 390
459 408 394 343 and then falls to clear at 21 (these traces, on F9's grid; the GPU there took 21 too). With the fused
sampler (15:09) its soft values moved enough to be given up at 9 (under the C's rule the GPU now clears it at 22),
and the full replay lost that one block (93,133 against the C rule's 93,134 on the same tree). So the constants were
set again with the margin over every frame. They save less: 47.0%, 49.4% and 43.6% of the
iterations of the 19 x 40 set, the hard runs and every frame, against the first set's 54.9%, 57.7% and 51.5%. No
recorded frame is left out of the choice now. The margin is what stands for a held-out set.

Rules measured and left. The first number pair is the blocks the C's rule verifies that each rule gives up (19 x 40
set / hard runs); the second is the iterations saved.
- The block's estimate (`BLK.x`, the soft stage's) under 0.7: 0 / 12 blocks, 54% / 61%. The 19 x 40 set verifies
  nothing under 0.74, but the hard runs verify 12 blocks between 0.55 and 0.7.
- The count after iteration 1 alone, at or over 0.30 m: 1 / 19 blocks, 65% / 68%. At 0.32 m: 0 / 4, 52% / 56%.
- No new low of the count in k iterations. k = 8: 4 / 21 blocks, 25% / 26%. k = 12: 0 / 3, 14% / 15%.
- A lower cap of 20: 127 / 395 blocks.
- The C's ratio test run earlier (at iteration 3, over 0.95 of the first count): 31 / 78 blocks, 32% / 30%.

On the sender's own symbols plus Gaussian noise (21-34's format, 512 codewords a level), the funnel verifies what
the C's rule verifies at every sigma from 0.55 to 0.8 (512, 511, 488, 388, 223, 53, 4, 0, 0 and 0), with 0 to 49%
fewer iterations. Near the code's threshold the failures are not hopeless, and the funnel leaves them alone.

Checked (2026-09-24, the tree after the sampler fusion):
- `scripts/gpu/back/test_ldpc.mjs` on SwiftShader, the iGPU and the 4090 (104 frames of four runs from the C's own grids, B 8): the
  GPU gives the funnel reference's verdict and iteration count on every block (4,640), 0 of the C's blocks given
  up, bad 0, counters and records right on every frame, zeroed frames nothing, the noise curve equal at every level.
  `decodeStall` under the funnel equals the fast copy on every frame it ran. Under `STOP=c` the GPU still equals the
  C on every block. The iterations a batch of 8 fall from 4,194 to 2,128 (02-36), 4,969 to 2,065 (21-34) and 2,608
  to 655 (12-00), and the iGPU's LDPC from 6.35 to 3.31, 7.47 to 3.21 and 4.02 to 1.05 ms a frame (i16, 12.4 us an
  iteration either way: the time follows the iterations).
- The 19 x 40 set on the 4090: 11,425 blocks under both rules, the per-frame dumps identical on every frame. The
  iGPU in f16: 11,435 under both, identical on every frame. LDPC iterations 179,492 to 94,731 (47.2% fewer).
- Every frame of every run on the 4090: 93,134 under both rules; one frame differs, 09-14-21 frame 487, by its
  finder score alone (0.461 against 0.472, the same 12 blocks), and it moves between two runs of the same rule too.
  No block given up on any run. Iterations 1,711,682 to 965,337 (43.6% fewer).
- Against the dumps from before the sampler fusion (13:36), 17 frames of the 19 x 40 set (net 0) and 153 of every
  frame (net +10) moved their blocks. Each of them moved the same way under the C's rule on the fused tree, so the
  moves are the fusion's.
- SwiftShader, `scripts/exp/gpu_captures.mjs` FINDER=cnn FRAMES=8 RUNS=20-04-09: builds and gives 190 blocks (the C 184).
  `scripts/exp/gpu_front.mjs` runs the wasm back half, so the funnel cannot move it; its cells were unchanged.
- The other row kernels on SwiftShader under the funnel (12-00 and 21-34, 8 frames each): `base` equals the reference
  on every block; `i16flags` gives the same verdicts and exits an iteration later on 39 of 768 blocks, as the flags
  variants do under the C's rule.

Time on the iGPU in f16, whole decode through `archive/build-scratch-2026-09/gpu_captures_ldpc.mjs` (a copy of `scripts/exp/gpu_captures.mjs` that
passes `LDPC_STOP` to the page), 48 frames, B 8, 2 in flight, `PROFILE=1`. The two rules ran back to back, twice,
16:22 to 17:15, on the tree with the fused sampler and the proposer as of 16:27. LDPC is the median of the two;
the frame and the rate are from the first pair, whose two runs had the same front half. Blocks are those runs'.

| run | the C's rule: LDPC, frame (ms), frames a second | the funnel | LDPC saved | blocks, both rules |
|---|---|---|---|---|
| 20-04-09 (1080, 1024/256) | 2.03, 10.01, 97 | 1.89, 9.89, 98 | 7% | 1,341 |
| 02-36-59 (1024/256, most blocks fail) | 6.13, 13.39, 73 | 3.04, 10.31, 94 | 50% | 96 |
| 21-34-51 (1440, 1024/512) | 8.00, 18.33, 53 | 3.01, 13.39, 73 | 62% | 1,745 |

Earlier today, before both front-half changes, the C's rule gave 2.01, 6.22 and 7.91 ms of LDPC (frames 10.63,
14.89 and 20.44 ms, 91, 66 and 48 a second), three runs each. The first constants gave 1.80, 2.19 and 2.55. In the
second pair the 1080 run's two sides straddled a proposer change and read 1,341 and 1,340 blocks. Run again back to
back, twice each, 20-04-09's 48 frames gave 1,340 under both rules, the same on every frame. The iGPU's LDPC time
follows the iterations run: 12.4 us an iteration at B 8 in the standalone test, under either rule. The 4090's times
are not reported (gpu/README.md).

## 7. CRC-32, compaction, the result record

On a clear syndrome the workgroup packs the k = 3816 systematic bits (L[j] < 0 is a 1) MSB first into 477 bytes
in the space L used (lane i packs bytes i, i + 106, ...). Lane 0 runs CRC-32 over bytes 0..472 bitwise (3784
steps, under one LDPC iteration of the workgroup; a 1 KB table would cost workgroup memory) and compares with
bytes 473..476 read as a big endian u32. Equal: verdict 1, `atomicAdd` on REC.count claims a slot, lanes 0..120
write the record's 121 words. Unequal: verdict 4. ITS[b] = iterations, or the C's codes: -2 declined, -3 stall,
-1 cap. A stalled or capped block writes verdict 3 and no record. The host reads count, count * 484 bytes, V,
ITS, BCOUNTS; the front half's counters ride in the same readback. The harness checks each payload after its
4 byte id against `sourceFor(spec.stream)(id)`: `bad` must be 0; a receiver has only the CRC, which is what
makes the CRC the last word. A frame that decoded nothing costs no readback bytes beyond its verdict row.

## 8. Tables and where each comes from

`M = await init()` (sim/ob.mjs), a `Focus(n, subch, 1, { span, bitmap })` built first; the wasm holds one
configuration at a time. Verified today at every format above (shapes printed from the exports). Exports this table
named then and gone since 2026-10-22: `_focus_block_tiers` (each block's tier; `_focus_tiers_for` gives a version's
profile), `_ob_test_crc_pack`, `_focus_spectrum_out` (`read()` gives the same coefficients) and `rxBits`.

| table | export (call) | shape |
|---|---|---|
| n, bw, vblocks, subch, blocks, npos | `M._focus_shape(p)`, `Focus.shape()` | 6 i32 |
| pos, decoded to (u, v) | `M._focus_pos_ptr()`, `Focus.posTable()`; v = (p div (n bw)) bw + (p mod (n bw)) mod bw, uw = (p mod (n bw)) div bw, u = uw < n / 2 ? uw : uw - n | subch * 320 u32; entry 0 is (1, 0) |
| cx, qx, X1, X2, sx2, sq2 | `M._focus_tables_out(p)`, `Focus.tables()` | 6 n + 2 f32 (1538, 3074, 6146, 12290) |
| block first sub-channel, count | `M._focus_block_subs(pFirst, pCount)`, `Focus.ldpcCode().first, .count` | blocks i32 each; first = 8 b, count 8 (at one rate; under the rate profile each block's own first sub-channel and its code's 7, 8, 9 or 12) |
| LDPC dims and schedule | `M._focus_ldpc_tables(tier, pDims, pLay)` (pLay = 0 sizes it), `Focus.ldpcCode().code` (tier 0: the profile's innermost code since 2026-10-07) | dims at 3/4: n 5088, k 3816, m 1272, z 106, zp 112, mb 12, kb 36, norm 13, slotsMax 17, slotsTotal 204, slots 180; lay 373 i32 |
| block bytes | the return of `M._focus_setup(...)`, `Focus.blockBytes` | 473 |
| bit map mode | `M._focus_bitmap_get()` | 3; 0 for a replay of a recording without `bitmap` (`recordedSpec`) |
| perm, inv, white, MAP | not exported; regenerated in JS from section 1 (`bitmap.mjs`) and verified against what the C paints (section 9). C to add: `int focus_bitmap_tables_out(int tier, uint8_t *white, int32_t *perm)` in src/wasm.c copying `F.ws->white` (subch * 640 B) and `F.ws->perm[tier]` (`code[tier].n` i32) | perm 5088; white subch * 640 |
| g0, step | `M._focus_grid_out(p)` | 2 f32 |
| CRC-32 | `M._ob_test_crc_hash(p, len)` | |
| decline bar, verdicts | `M._ob_test_crc_gate_out(est, blkEst, bar, declined)` | blocks each |
| truth stream | `streamBlock(id, len)` (sim/ob.mjs, `_stream_fill`); `sourceFor(spec.stream)` (sim/phy.mjs) | |
| the C's spectrum | `Focus.measure(true)` then `read()`: 2 * 320 * subch f32, the detrended coefficients in pos order | |
| the C's verdicts | `Focus.rxFinish().ok`; `Focus.blockStats()` its (-2 declined, -3 stall, -1 cap, > 0 iterations) and est | blocks |
| a grid from the C's registration | `M._ob_test_sample(img, iw, ih, 1, 2, dims, state, grid)` | n x n f32 row major, pitch n |
| finish from soft values, verdicts | `Focus.llrView()` + `rxLdpc(est)`; `rxAssemble(bytes, stride, verdicts, its, est, quad)` | |
| the exact symbols sent | `Focus.measure(true)`, `encode(blocks)`, `sent()` | 640 subch int8, +1 / -1 |

Missing exports: `focus_bitmap_tables_out` (recommended: makes the regeneration checkable without an encode).
Optional: `int focus_llr_out(int8_t *llr, float *est)`, the C's own slot order LLRs after a finish (focus.c reuses
one block's worth at a time, so it would have to keep subch * 640 bytes), and `focus_ldpc_encode_out(tier, data,
cw)` for synthetic curves. Neither is needed: the JS reference in section 9 stands in for the first and the JS
encoder for the second, both checked against the C.

## 9. Standalone tests against the wasm, no GPU front half

Archived since 2026-10-22 (`archive/gpu-back-tests/`, with the one-rate kernels and the grid pass 1 they held); the
record of how they were built and what they read follows.

Shared loader `scripts/gpu/back/frames.mjs`: for a run under `research/captures` and a frame list, read `meta.json` and the
`.gray` frames (`meta.sizes[i]` or `meta.w, meta.h`), `spec = recordedSpec(meta.config.spec)`, `phy = await
makePhy(spec)`, `fc = new Focus(spec.n, spec.subch, 1, { span: spec.span ?? 0, bitmap: spec.bitmap })`.
Per frame: (a) the grid: `_ob_test_sample` (the C's own registration and sampler) reordered to strips,
strip[(x div 64) n 64 + y 64 + x mod 64] = grid[y n + x]; (b) the C's finish on that grid: `rxAcquireQuadNoImage(64,
64, [8, 8, 56, 8, 56, 56, 8, 56], 0, 1, { mesh: 0 })` as `harness/score.mjs` does, `measure(true)`, `gridView(n).set(strip)`,
`d = rxFinish()`, then `read()`, `blockStats()`; (c) `phy.tally({ ...d, found: 1 }, 0, 0)`
for seen and bad against the truth stream. Checked today: this path decodes the same blocks as the one call
decode on 24 of 24 frames (12 each of 02-36-59 and 21-34-51), and `read()` equalled the C's spectrum export of the time at every
pos (0 error). The JS encoder in `bitmap.mjs` (H from `lay`, the CRC, perm, white) against `sent()`: 0 of 163840
slots differ at 1024/256 bit map 3, 0 of 10240 at 256/16 bit map 3, 0 of 81920 at 512/128 bit map 0. n = 2048 has
no recording: frames come from `scripts/gpu/harness/scenes.mjs` `renderScene(cell, phy.frame(i), i, pxm)` and are
reported as simulated.

GPU side: `runGpu` from `scripts/gpu/harness/run.mjs` (imported, not edited) with `page: "/scripts/gpu/back/test_page.html"`,
an own page under `gpu/back/` that loads one stage module, takes a
job of tables and input buffers, runs one batch of B frames inside a timestamp pair, posts the output buffers
and the GPU ms back. Every run under the quiet machine lock, `IGPU=1` (f16 variant) and `HWGPU=1` (f32), the
adapter line printed, `pgrep -f lizard-gpu` empty first. A time is GPU ms a frame at the stated B, blocks beside.

JS reference `scripts/gpu/back/ref_soft.mjs`: section 1's B arithmetic in f64 from `read()`'s coefficients, giving L in
slot order, est and declined; equal to the C's own by construction (double, the same rounding), and checked by
`rxLdpc` giving `rxFinish`'s blocks. It is what B is held to and what A and C feed the C with.

- **A** (`scripts/gpu/back/test_transform.mjs`): grid in; S out. Error: rms of n^2 S minus the C's coefficients over the disc,
  relative to their rms, in dB. Blocks: `ref_soft(S)` into `llrView()` + `rxLdpc(est)` against `rxFinish().ok`.
- **B** (`scripts/gpu/back/test_soft.mjs`): S built on the host from `read()` laid into the disc order; L, EST, BLK out. Slots
  equal to the reference, the rest off by one; declined equal to `_ob_test_crc_gate_out`; blocks through `rxLdpc`.
- **C** (`scripts/gpu/back/test_ldpc.mjs`): L from the reference (or B's), EST; V, ITS, REC out. REC is compact, so the test lays its
  payloads into a per block array first, then `rxAssemble(bytes, 473, verdicts, its, est, quad)` against
  `rxFinish().ok` and `blockStats().its` block by block; `phy.tally` for bad. Since the stop rule (section 6), the
  GPU is held block by block to `ref_ldpc.mjs`'s decoder under the same rule. Under `"c"` that decoder must equal
  `rxLdpc` on every block. Under the learned rule (the funnel until 2026-09-28), the blocks `rxLdpc` verifies and
  the rule gives up are counted as the rule's cost. A synthetic curve: `phy.frame(i)`
  encoded, `sent()` gives the exact symbols, Gaussian noise added in JS at five levels, LLRs from the reference,
  blocks decoded by `rxLdpc` and by the GPU at each level.

## 10. Three stages: files and acceptance

As planned and built on 2026-09-24. Since 2026-10-07 B's and C's kernels are the rule stage's (`tiers.mjs`,
`wgsl/back_tiers.mjs`); `wgsl/back_soft.mjs`, the one-rate LDPC source and the tests named here went on 2026-10-22.

Files: host `gpu/back/<stage>.mjs`, shaders `gpu/wgsl/back_<stage>.mjs`, test `scripts/gpu/back/test_<stage>.mjs`. Each
host module has `build(device, { B, sizes, tables })` creating its outputs and pipelines and `encode(enc, lane)`
recording its passes; the next stage binds the previous one's buffers by the names in section 2. Bind group
layouts explicit; every function scope `var` initialised; no `pass`, `meta`, `ref`, `shared`, `out` as names;
no em or en dashes anywhere; comments say why.

- **A: detrend and transform.** `transform.mjs` (gate and LISTS/ARGS, A1, reduce, A2, `discLayout(shape, pos)` giving
  V, off, umax, UV, used by B), `wgsl/back_transform.mjs`, `scripts/gpu/back/test_transform.mjs`, plus the shared `scripts/gpu/back/frames.mjs`,
  `scripts/gpu/back/test_page.html`, `scripts/gpu/back/test_page.mjs`, and `back.mjs` (the composition behind the front half: binds `gridBuf`,
  `gridStride`, `selBuf`, `framesBuf`, records A, B, C in the front half's encoder, reads REC, V, ITS, BCOUNTS).
  In: P, sel, frames. Out: LISTS, ARGS, PART, COEF, Y0, Y, S, BCOUNTS.
- **B: soft values, bit map, whitening, decline gate.** `soft.mjs`, `wgsl/back_soft.mjs`, `scripts/gpu/back/test_soft.mjs`,
  `scripts/gpu/back/ref_soft.mjs`, `bitmap.mjs` (perm, inv, white, MAP for a format and bit map mode, the JS encoder, and the UV
  table from `discLayout`). In: S, LISTS, ARGS. Out: L, EST, BLK.
- **C: LDPC, CRC, compaction.** `ldpc.mjs`, `wgsl/back_ldpc.mjs` (one shader), `scripts/gpu/back/test_ldpc.mjs`. In: L, BLK, LISTS,
  ARGS, MAP and the LDPC tables. Out: V, ITS, REC.

Acceptance. Runs (`research/captures`), with the C's one call decode counted today, first 60 frames and all 600:
12-00 = `run-2026-09-22T12-00-29-891Z` (1024/256, bit map 0: 960 / 12872 blocks), 21-34 = `run-2026-09-23T21-34-51-057Z`
(1024/512, bit map 3: 2283 / 25690), 02-36 = `run-2026-09-22T02-36-59-424Z` (1024/256, bit map 0: 128 / 2314,
the hard run, most blocks fail), 06-26 = `run-2026-09-23T06-26-06-874Z` (256/16, bit map 3: 120 / 585, found 591
of 600). "The C" is `rxFinish` on the same grid, which equals those counts (section 9). Quick = the first 60
frames; `FULL=1` = all 600. Times are GPU ms a frame at B = 8, INFLIGHT 2; iGPU f16, 4090 f32; this desktop's
wasm on the same frames is transform 1.14 / 1.38 ms, detrend and LLR 0.36 / 0.52, LDPC 4.13 / 5.43 on 02-36 / 21-34,
and the 2026-09-22 port on the iGPU was 4.2 ms a frame for its transform chain and 8.5 for its LDPC (96 real frames).

| who | equality, on 12-00 and 21-34 unless said, quick and FULL | iGPU | 4090 |
|---|---|---|---|
| A | S within -40 dB of the C (f16), -100 dB (f32); blocks through the reference and rxLdpc at least 99% of the C's; 0 bad | 1.5 ms at 1024/256, 2.5 at 1024/512, 6 at 2048/1024 (simulated) | 0.10, 0.15, 0.4 |
| B | 99.9% of slots equal, the rest off by one; declined equal on every block; blocks through rxLdpc at least 99% of the C's; 0 bad | 0.3 ms at 1024/512 | 0.03 |
| C | from the reference LLRs, decisions and iterations equal to the C's on every tried block, blocks equal, also on 02-36 and 06-26; 0 bad; the synthetic curve equal to rxLdpc at every noise level | 3 ms on 02-36, 1.5 on 12-00 | 0.3 on 02-36 |
| all | behind the front half on all 19 runs: at least 99% of the 92 247 blocks the front half gets through the wasm back half today, 0 bad; the negative control (a decoded frame zeroed) gives 0 | 4 ms at 1024/256 | 0.5 |

The time budgets are what the traffic and operation counts above allow with margin; a miss is reported as a
number with the profile beside it. Whatever a test reads back for comparison (S, L, decisions) is a test path
only; the product reads back REC, V, ITS and BCOUNTS and nothing else.

## As built: A (2026-09-24)

Nothing above was found wrong against the C; these are the details B and C bind to, from `gpu/back/transform.mjs`.

- `Transform.build(device, { B, tables, precision, cpw, tw })` with `tables = await transformTables(sizes)`, sizes in
  ascending n (up to 4 then, `SLOTS` = 6 since 2026-09-27; a size's index is its position, and `sel[f].z` indexes it). `t.lane({ gridBuf, gridStride,
  framesBuf, selBuf })` returns the lane; `t.encode(enc, lane, { ts })` records gate, pass 1, reduce, pass 2 as
  four compute passes (a timestamp pair each when `ts = { qs, base }` is given).
- Per frame strides are the largest size's in the build: `t.sStride` (S entries), `t.yStride` (Y entries, V n
  since 2026-09-25), `t.partStride` (= nmax; the column means ride in PART's w, so there is no Y0 since 2026-09-25);
  `t.cbytes` is 4 (f16x2) or 8 (f32x2). Frame f's disc starts at
  `f * t.sStride` entries into `lane.sBuf`; row v of it at `tables[size].off[v]`, entry e of the row is u = e for
  e < np (u = e + 1 on v = 0), else e - np - umaxN, with `rows[v] = (off, count, np, umaxN)`. `tables[size].UV[i]`
  is the entry of pos[i]. The rows table is bound as a uniform (V <= 512) so pass 2 holds to 8 storage buffers.
- `lane.listsBuf`: u32 [SLOTS][1 + B], count then frame indices, size major. `lane.argsBuf`: u32 [SLOTS][12] (pass 1,
  reduce, pass 2, then since the composition B's blocks slot (blocks, 1, count)) with usage INDIRECT, entry k of
  size s at byte offset 4 (12 s + 3 k) (`ARGS_SLOT`, `argsOffset`); a later stage that dispatches one workgroup a
  frame of size s can reuse the reduce entry, (1, 1, count). The gate lists a frame only when frames[f].valid and
  sel[f].y >= 2 and sel[f].z names a built size below `SLOTS` (4 then); it used to fold an index of 4 or more into
  size 3.
- `lane.coefBuf`: f32 [B][8] = (px, py, pxy, pxx, pyy, 0, 0, 0). `lane.countsBuf`: u32 [B][8], cleared by
  `encode` (or, built with `zero`, by the gate, which also zeroes C's V, ITS and REC count: the composed chain),
  columns 0 to 2 written by A (columns transformed = n, rows transformed = V, entries stored = npos); B and C take
  columns 3 to 7. `lane.bytes` is the lane's device memory.
- The stored S is T / n^2 as designed. In f16 the smallest disc coefficients of a blurred frame land in f16's
  subnormals (below 6.1e-5 in S units); the rms error over the disc is what section 9 measures, and it is set by
  the strong coefficients. Should a scale ever be wanted, it is one constant at pass 2's store and B is scale free.
- Pass 1's shape differs from section 3 in one respect: a workgroup's four columns come in as one vec4f load a
  strip row and the quad's two pairs are transformed side by side (two exchange buffers, one set of twiddles and
  barriers), so Y(v) for the four columns leaves as one 16 B store (f16, `vec4u` of `pack2x16float`) or two
  `vec4f`. The scalar version (one 4 B load a column a row, pairs in turn) put pass 1 at 95% of the iGPU's time:
  1.23 of 1.30 ms a frame at 1024/256 against pass 2's 0.06 with half the FFTs. Widening the tile to 8 columns
  did not help (1.62 ms scalar, 0.75 vec4, against 1.30 and 0.60 at 4), nor did the twiddle table (1.40) or
  sincos (1.28, and -118 dB on the 4090 against the chain's -129); the cost was the number of cache lines a wave's
  load touches, not what it fetches. Workgroup memory: two exchange buffers plus 48 B a thread of partials, 14 KB
  at 1024 f16, 28 KB at 2048 f16 (phones), 45 KB at 2048 f32 (the 4090's 48 KB only; SwiftShader's 32 KB cannot
  build it, so the test simulates 2048 on real adapters only). At 1536, the largest picture since 2026-09-27,
  21,504 B in f16 and 33,792 B in f32, which SwiftShader's 32 KB cannot build either.
- Measured (`scripts/gpu/back/test_transform.mjs`, 16 frames a run registered by the C's own sampler, B = 8, 5 submits a batch,
  the median; blocks = the disc through the soft-value reference and the C's LDPC, the C's own count beside):

  | device, precision | 1024/256 (20-04-09) | 1024/512 (21-34-51) | 256/16 (06-26-06) | 2048/1024 (simulated 2160) |
  |---|---|---|---|---|
  | iGPU f16, ms a frame (gate + pass 1 + reduce + pass 2) | 0.60 (0.001, 0.538, 0.001, 0.060) | 0.85 (0.001, 0.762, 0.001, 0.090) | 0.075 | 3.64 at B = 2 (3.37 pass 1) |
  | iGPU f16, S vs f64 dB (worst frame) | -60.0 (-58.1) | -59.4 (-58.4) | -63.2 (-61.6) | -59.9 (-55.9) |
  | iGPU f16, blocks GPU / C, bad | 390 / 389, 0 | 568 / 568, 0 | 32 / 32, 0 | 256 / 256, 0 |
  | 4090 f32, S vs f64 dB | -129.5 | -129.0 | -131.9 | -128.7 (1024 / 1024 blocks) |
  | iGPU f32, ms a frame | 0.96 | | 0.088 | |
  | iGPU budget (section 10) | 1.5 | 2.5 | | 6 |

  The C's f32 spectrum sits -129 to -133 dB from the same f64 reference, so the f32 GPU is the C's equal and f16
  is 70 dB under the strongest ring. Counters exact on every frame. Pass 1 is still 90% of the iGPU's time; its
  remaining cost is the strided read and the Y store, both of which the sampler fusion removes.
- What the fused sampler must provide, seen from pass 1 as built: (1) a WGSL function `samplePicture(f: u32,
  size: u32, x: u32, y: u32) -> f32` importable as a string, reading module (g0 + x step, g0 + y step) of frame f
  through the map, the Coons residual and `bilin0`, as F9 does; pass 1 calls it 8 x 4 times a thread (its 8 rows
  of a quad) in place of the vec4f load and needs nothing else from the picture. (2) Its bindings added to pass 1's
  bind group layout (the texture array and its sampler, `maps`, `sizes`, `resid`, `lattice`, `LT`) beside the eight
  buffers there now; pass 1 then has 8 storage buffers plus a texture, a sampler and uniforms, within the default
  limits. (3) `sel[f]` and `frames[f].valid` as they are: the gate already lists frames by `sel[f].z` and skips
  `sel[f].y < 2`. With that, `gridBuf`, the strip order and F9's pass disappear; PART, Y0, Y and everything after
  are unchanged, and the picture's 4 MB a frame at 1024 is never written or read.

## Corrections and as built: B (2026-09-24)

Sections 1, 4 and 5 hold against the C. The references behind the test: the JS encoder (`bitmap.mjs`, H from
`lay`, CRC-32, perm, white) against `sent()` differs on 0 of 327 680 slots at 1024/256 bit map 0, 0 of 655 360 at
1024/512 bit map 3, 0 of 20 480 at 256/16 bit map 3; `refSoft(read())` (`scripts/gpu/back/ref_soft.mjs`, the C's arithmetic with
its float and double steps) finished by `rxLdpc` gives `rxFinish`'s verdicts and iteration counts on every frame
checked (16 of 12-00, 16 of 21-34); A's `scripts/gpu/back/ref_transform.mjs` sits -129 to -133 dB from the C's f32 spectrum. What
differs from the text, and the interfaces C and the composition bind to (`gpu/back/soft.mjs`):

- One workgroup a BLOCK, not a sub-channel: 256 threads = 8 sub-channels x 32 lanes x 10 coefficients. Same gather
  and pack as section 4 (a lane's 10 coefficients are 20 slots, five u32 words), and the block's estimate and gate
  come out of the same workgroup, so BLK needs no atomic count of 8.
- Dispatch is direct, `(blocks_s, 1, B)` a served size, the frame on z as A's passes take it, and the kernel reads
  the size's frame count from LISTS and returns before its first barrier when z is past it. `build(..., { indirect:
  true })` switches to `dispatchWorkgroupsIndirect` from A's blocks slot of ARGS, `(blocks_s, 1, count_s)` at
  `ARGS_SLOT.blocks` (the gate writes it since the composition).
- `build(device, { B, sizes, tables, precision, indirect, cap })`: `sizes[s] = { n, subch, blocks }` at the size's
  index (`sel[f].z`), `tables[s].uv` or `.UV` (A's `discLayout` and `bitmap.mjs discLayout` build the same table).
  Every stride is `dims.mjs derived(sizes, { B, cap })`'s, the one function C and the composition size themselves by
  too (before it, C recomputed its own from its size list). `stage.lane(ln)` reads `ln.S`, `ln.LISTS`, `ln.BCOUNTS`
  (and `ln.ARGS` when indirect) and adds `ln.L`, `ln.EST`, `ln.BLK`; `stage.encode(enc, ln, { timestampWrites })` is
  one compute pass and `stage.dispatch(pass, ln)` the same dispatches in a pass the caller holds. Strides are the
  largest served size's: L `subchMax * 160` words a frame (int8 slot order, `llrView`'s layout), EST `subchMax` f32,
  BLK `blocksMax` x `vec2<u32>` = (bitcast f32 estimate, declined). BCOUNTS column 6 = sub-channels quantised (8 a
  block); columns 3 to 5 and 7 are C's.
- Files: the frames loader is `scripts/gpu/back/frames.mjs` and the page `scripts/gpu/back/test_page.mjs` through `scripts/gpu/back/spawn.mjs`, shared by A, B, C and the
  chain since 2026-09-24 (they were `frames_soft.mjs`, `soft_page.*` and `gpu_run.mjs`, copies of A's; the calls are
  unchanged). `scripts/gpu/back/ref_transform.mjs` is A's, and B's test runs on it (`refDisc(refSpectrum(strip, n, V), pos, n, bw)`).
- Measured (`scripts/gpu/back/test_soft.mjs`, 16 frames each of 12-00 and 21-34, B = 8, S from the C's `read()` and from the f64
  reference alike, the C's own grid): f32 on every device (SwiftShader, the 4090, the iGPU) 99.999% of slots equal
  the reference, the rest off by one, sign agreement 100.0000% where both are nonzero, rho 1.000000, declined equal to
  `_ob_test_crc_gate_out` on every block, `rxLdpc` from the GPU's L gives every block `rxFinish`'s verdict on
  32 of 32 frames (288 + 568 blocks, 0 bad). f16 S on the iGPU: 99.51% (12-00) and 99.43% (21-34) of slots equal
  the C, the rest off by one, 0 by more, sign 100.0000%, rho 0.999999, 288 / 288 and 569 / 568 blocks, 0 bad. At
  T / n (S kept out of f16's subnormals) the slots that cross zero against the C fall from 41 and 100 to 1 and 4
  of 1.3 M and 2.6 M, blocks the same: one constant at A's pass 2 store if it is ever wanted. GPU ms a frame, one
  compute pass, timestamps, median of 10 repeats: iGPU f32 0.06 / 0.13 at 1024/256 / 1024/512, f16 0.05 / 0.11
  (the budget was 0.3 at 1024/512). Indirect dispatch from an ARGS slot changes
  none of these by more than 3%, so the direct dispatch stays. FULL (all 600 frames of each, iGPU, T / n^2): f32
  12872 / 12872 and 25690 / 25690 blocks from the C's S (12872 and 25689 from the f64 reference's); f16 12871 /
  12872 and 25689 / 25690 (12871 and 25693 from the reference's); 84 of the 197 M slots of 21-34 off by more than
  one in f16 (none of 12-00's), declined differs from the C on 1 of 57 600 blocks (est within 2.8e-5 of the bar);
  0 bad throughout. The table's "verdicts = C" column is `T.okSame`, a count of FRAMES on which every block's
  verdict equals the C's, not of blocks: 600 and 600 of 600 in f32 from the C's S (600 and 599 from the
  reference's), 567 and 502 of 600 in f16. So in f16, 131 of 1200 frames have at least one block whose verdict moved,
  and the gains and losses nearly cancel in the totals. An earlier reading here took that column for iteration
  counts; that run did not tabulate iterations (`scripts/gpu/back/test_soft.mjs` now prints both per frame).

## Corrections and as built: C (2026-09-24)

Sections 6 and 7 hold against the C: `ref_ldpc.mjs` (ldpc_decode_stall and the block tail in JS, integer for
integer) gives `rxLdpc`'s verdicts, iteration counts and payloads on every frame it was run on, and the GPU gives
the same (the numbers are at the end of this section and in `scripts/gpu/back/test_ldpc.mjs`'s table). What differs from the
text, and the interfaces the composition binds to (`gpu/back/ldpc.mjs`):

- **V and ITS are one word a block**, u32 verdict and i32 iterations, at `f * blocksMax + b` (WGSL has no byte
  stores; 1 KB a frame at 128 blocks). Verdict 0 not run, 1 verified, 2 declined, 3 stall or cap, 4 CRC; ITS the
  C's codes (-2 declined, -3 stall or given up by the stop rule, -1 cap). `encode` clears both and REC's count
  before the pass; in the composed chain the gate zeroes them, so `dispatch` can go into a pass the caller holds
  open.
- **REC** is u32 count then 484-byte records at 4 + 484 r: word 0 = frame + 65536 block, word 1 = iterations as
  i32, words 2 to 120 the 473 payload bytes then 3 zero bytes. `unpackRecords` lays them out a block for
  `rxAssemble`. Cap = B x blocksMax records; the count is claimed by one `atomicAdd` a verified block.
- **One dispatch, direct**: (blocksMax, B, 4), workgroup (x, y, s) block x of frame LISTS[s][1 + y]; a workgroup
  past its size's block count or frame count returns before its first barrier (LISTS and BLK are read-only
  storage, which the uniformity analysis accepts as uniform at a uniform index). No ARGS slot is read. (From
  2026-09-28 the decoder's pass one dispatched it from the gate's ARGS words, `argsLdpc`, over `SLOTS` sizes; since
  2026-10-07 the rule stage's LDPC runs pass one, a dispatch a code; the gate's words went on 2026-10-22.)
- **MAP** is built by the stage (`bitmap.mjs mapTable`) once a bit map MODE in play, with the largest block count
  that mode serves: block b's row is the same in every format (the whitening runs over the frame's slots in pos
  order, block b at 5120 b). u16 pairs packed in u32; a size's (blocks, MAP offset) rides in the params uniform.
- **The CRC-32 is split over 119 lanes**, not run bitwise on lane 0: lane w runs the reflected CRC over its 4-byte
  chunk from a zero state, multiplies by x^(8 * bytes after it) mod P (a 32-step carry-less multiply against a
  120-word table in the uniform, `crcPowers`), XORs into one workgroup atomic; the initial state's term and the
  final inversion are one folded constant. Checked against `crc32` on random 473-byte messages.
- **Workgroup memory 31 024 B**: L as i32 [5088], R as vec2<u32> [1272], the 120 packed words, two atomics and
  a flag. Under the 32 768 the phones report; the page asks for `maxComputeWorkgroupStorageSize` (the default
  16 384 would refuse it). The exact syndrome runs every iteration (the flag variant of section 6 was not built).
- **BCOUNTS**: column 3 blocks tried (not declined), 4 verified, 5 records written, 7 iterations run (summed
  over the frame's codewords).
- `build(device, { B, sizes, code, cap, variant })`: `sizes[s] = { n, subch, blocks, bitmap }` at the size's index,
  `code` = `Focus.ldpcCode().code` (its `lay` is what H is rebuilt from; the shader is written to 12 rows of 15
  data slots and `layWords` checks it). lStride, blocksMax and the record cap are `dims.mjs derived(sizes, { B, cap
  })`'s, as B's are (C took lStride from B and derived blocksMax from its own list before). `stage.lane(ln)` reads
  `ln.L`, `ln.BLK`, `ln.LISTS`, `ln.BCOUNTS` and adds `ln.V`, `ln.ITS`, `ln.REC`; `stage.encode(enc, ln, {
  timestampWrites })` is three clears and one compute pass, `stage.dispatch(pass, ln)` the dispatch alone.
- Files: C's test runs on the shared `scripts/gpu/back/test_page.mjs` (job kind "ldpc") through `scripts/gpu/back/spawn.mjs`, with `scripts/gpu/back/frames.mjs`,
  `scripts/gpu/back/ref_soft.mjs` and `bitmap.mjs` (C had its own `ldpc_page.*` and B's `gpu_run.mjs` before 2026-09-24). The page
  fetches a stage job only once the one before it is scored: `scripts/gpu/back/test_ldpc.mjs` and `scripts/gpu/back/test_soft.mjs` close a run's wasm
  handle when their generator moves on, and the merged page's prefetch of two had them scoring a run's last jobs on
  a closed handle (seen at B = 32: a sweep point read 0 blocks against 568 records).
- **Variants** (`wgsl/back_ldpc.mjs VARIANTS`, `VARIANT=` in the test): `base` as section 6 describes it; `hoist`
  keeps a row's 17 indices and q values in registers so pass 2 reads no workgroup memory; `reg` is hoist with
  the minima tracked by select (the C's SIMD kernel's form); `i16` is reg with the posteriors as two i16 a word,
  20 848 B of workgroup memory, a write being one atomicXor of old ^ new on the lane's own half; `flags` and
  `i16flags` gate the exact syndrome on the rows' own parities. **`i16` is the default**: the fastest of the
  variants whose decisions and iteration counts equal the C's, and the one that leaves a 32 KB phone room.
  The stop rule is a separate choice (`STOP=` in the test, section 6's "The stop rule"). The numbers below are all
  under `"c"`, the C's rule.

Numbers (2026-09-24). Frames through the C's own sampler and finish; soft values `refSoft(read())`; the C's
verdicts from `rxLdpc` on those (equal to `rxFinish` on all 104 frames); 12-00 16 frames, 21-34 56, 02-36 16,
06-26 16 (4640 blocks). Every device (SwiftShader, the 4090, the iGPU) and every variant but the two flags ones:
verdict and iteration count equal to the C's on 4640 of 4640 blocks, 2513 blocks verified = the C's, 0 bad,
counters and records right on 104 of 104 frames; the zeroed frames verify nothing (every block verdict 4, as the
C says of them); the synthetic curve at 21-34's format, sigma 0.45 / 0.55 / 0.62 / 0.68 / 0.8 an axis, 512 / 512 /
488 / 0 / 0 blocks on 8 frames each, equal to `rxLdpc` in verdicts and iterations at every level; the JS
reference equal to the C on 32 of 32 frames. The flags variants: verdicts and blocks equal, 0 bad, the exit one
iteration later on 36 of 512 (12-00) and 135 of 1024 (21-34) blocks. Iterations of the verified blocks: 12-00
mean 1.7 (151 at 1, 88 at 2, 34 at 3, 15 above); 21-34 mean 3.2 (634, 640, 355, 170, 96 at 1 to 5, a tail to 30);
02-36 1.2; 224 of 512, 1422 of 3584 and 480 of 512 blocks stall or cap on the three, which is where the time
goes: 2300 to 7500 iterations a batch of 8.

Time, GPU timestamps a batch of 8 (median of 10 timed passes after 2 warm passes), ms a frame, on 02-36 /
12-00 / 21-34 (16 frames each), and the us a codeword-iteration the sweep settles at:

| variant | iGPU (RDNA-2, 2 CUs) ms a frame | us an iteration |
|---|---|---|
| base | 11.0 / 7.0 / 13.0 | 21 |
| hoist | 7.2 / 4.6 / 9.4 | 14.1 |
| reg | 7.2 / 4.6 / 9.3 | 13.7 |
| i16 (default) | 6.3 / 4.0 / 8.2 (7.5 over all 56) | 12.1 |
| flags | 5.6 / 3.6 / 7.5 | 11.0 |
| i16flags | 4.6 / 3.0 / 6.2 | 9.0 |

Scaling with codewords in flight. iGPU, i16, 02-36: 32 codewords 6.97 ms (477 iterations, 14.6 us each), 64
6.99 (13.6), 128 20.4 (12.6), 256 50.8 (12.1); 21-34 at B = 32, i16: 16 frames in flight (1024 codewords, 10 442
iterations) 7.87 ms a frame, 12.1 us an iteration, and 32 in flight (2048 codewords) 8.03 ms a frame, 11.9 us, blocks
equal to the C's (`IGPU=1 B=32 RUNS=21-34:32 VARIANT=i16 REPEATS=3`). The 13.7 ms a frame this line gave before was that point measured
with VARIANT=base (21 us an iteration), not i16. Time is proportional to iterations run from 64 codewords up: nothing is left to hide, the WGP is
instruction bound (six resident codewords in place of four, i16 against reg, bought 1.13x; the flag gate,
which removes a third of the reads and the index arithmetic, 1.27x). Budgets: the iGPU misses 3 ms on 02-36 and
1.5 on 12-00 by 2.1x and 2.7x with the default, 1.5x and 2x with i16flags, and sits beside this desktop's wasm
(4.13 / 5.43 ms on 02-36 / 21-34). The 4090's times are not reported (its LDPC times were bimodal between runs of
the same test on the same batch, by 5x to 12x; it idles at 210 MHz between the test's jobs). The iGPU shares
package power with a CPU other jobs were loading (load average 10 to 28 during these runs); an iGPU submit of about 2 s or
more (base at B = 32, 5 timed passes: 3 s) comes back empty with no error. Not built: the two codewords a workgroup variant, and R packed at 36 bits a check (9 words for
8 checks), which with i16 posteriors would be 15.9 KB, two codewords in a 32 KB phone core.

## 11. As composed (2026-09-24)

`back.mjs` is the chain behind the front half. `BackHalf` owns A, B and C, their tables, and one set of per-frame
strides from `dims.mjs derived(sizes, { B, cap })`, which `soft.mjs` and `ldpc.mjs` also size themselves from. B
and C each worked out their own strides before this, from their own size lists, and could disagree on where frame
f's slots sit. `scripts/gpu/back/test_back.mjs` ran the chain on the device as the decoder does (archived 2026-10-22).

### What the wiring calls

```
const bh = await BackHalf.build(device, { B, sizes, precision, cap, variant, log });
const ln = bh.lane({ gridBuf, gridStride, framesBuf, selBuf });   // a front-half lane's own buffers
bh.encode(enc, ln, { ts: { qs, base } });   // six compute passes, a timestamp pair each at qs[base + 2k]
bh.encode(pass, ln);                        // or the six dispatches appended to a pass the caller holds open
const used = bh.readback(enc, ln, readBuf, off);   // REC, V, ITS, BCOUNTS (and PILOT since 2026-09-30) copied to readBuf at off; used = bh.readBytes
const r = parseReadback(mappedArrayBuffer, off, bh.dims);
//   r.records: [{ frame, block, its, id, payload (473 B, id first) }], the verified blocks only
//   r.frames[f]: { verdict, its (blocksMax each), counts (COUNT: columns, rows, stored, tried, verified, records,
//   subch, iterations), pilots (a block's pilot reading, 2026-09-30), shift (the grid's, 2026-10-01) }
bh.destroyLane(ln);
```

- `sizes`: the decoder's six (four until 2026-09-27), at the size index `sel[f].z`, each `{ n, subch, span, bitmap }` or null. If `bitmap`
  is left out, the format's own is used (3, `focus_setup`'s). A replay must pass `recordedSpec`'s value: 0 for a
  recording made before 2026-09-23. With the wrong bit map, every block fails its CRC and nothing else goes wrong.
- Pass null for a size whose picture does not fit the lane's grid (n^2 > gridStride). The finder cannot pick such a
  size, and leaving it out sets blocksMax, and with it every stride and the readback, by the largest size that can
  occur. With 2048/1024 built, blocksMax was 128; with 1024/560 as the largest, 70. With 1536/1024 built (the largest
  since 2026-09-27) it is 128, in 8-sub-channel chunks (the rate profile's 121 blocks fit under it).
- A size whose pass 1 needs more workgroup memory than the device offers is left unbuilt, and the build logs it
  (1536 in f32 on SwiftShader's 32 KB; 2048 before 2026-09-27). The gate leaves a frame off every list when:
  - its size is not built;
  - its `sel[f].z` is `SLOTS` (6; 4 until 2026-09-27) or more;
  - its `sel[f].y` is under 2 (not found, or no version from its word or the held one);
  - its `frames[f].valid` is 0.
- The host clears nothing. The gate zeroes BCOUNTS, V, ITS and REC's count at the head of the chain, so the chain can
  go inside an open pass. The back half does not need the front half's grid clear.
- `lane()` binds the grid buffer it is given. Call it again, and `destroyLane` the old lane, whenever the front half
  replaces `gridBuf`: `FrontHalf.ensure` makes a larger one when nmax grows.
- `BackHalf.build` reads its tables from the wasm (`transformTables`). The wasm holds one configuration at a time,
  so build the back half while no other `Focus` is in use.
- `precision: "f16"` keeps Y and S in f16 where the device has `shader-f16` and falls back to f32 with a log.
  `variant` is the LDPC row kernel, default `"i16"`. `cap` is the number of records a batch (default 0, meaning B x
  blocksMax). A verified block past the cap is still counted: BCOUNTS column 4 against column 5.
- Device limits: workgroup memory at the adapter's maximum. The LDPC needs 20 848 B (21,384 B with the learned stop
  since 2026-09-28; the rule stage's 1/2 code with its stop 31,748 B). Pass 1 needs 14 KB (f16) or 28 KB (f32) at
  1024, and 28 KB or 45 KB at 2048 (21,504 B or 33,792 B at 1536, the largest since 2026-09-27). Each stage binds at
  most 8 storage buffers. `FrontHalf.create`
  already asks for both limits and for `shader-f16`.

### A lane's buffers

Strides are the largest built size's. B = 8. cbytes is 4 for f16 and 8 for f32.

| buffer | written by, read by | bytes |
|---|---|---|
| grid, frames, sel | the front half's; read only | (not the back half's) |
| LISTS | gate; every stage | 16 (1 + B) |
| ARGS | gate; indirect dispatch of pass 1, reduce, pass 2 (and B with `indirect`) | 192 |
| PART, COEF | pass 1, reduce; reduce, pass 2 (PART's w is the column mean, once Y0) | (16 nmax + 32) a frame |
| Y | pass 1; pass 2 | cbytes V n a frame (V - 1 rows written by pass 1, one of scratch for the cancel stage): 1.38 MB at 1024/560 f16, 3.74 MB at 2048/1024 f16 |
| S | pass 2; soft | cbytes 320 subch a frame: 0.72 MB at 1024/560 f16, 1.31 MB at 2048/1024 f16 |
| L, EST, BLK | soft; LDPC (EST is for tests only) | (640 subchMax + 4 subchMax + 8 blocksMax) a frame |
| V, ITS | LDPC; host | 4 blocksMax each a frame |
| REC | LDPC; host | 4 + 484 recCap a batch |
| BCOUNTS | gate (zeroes), A, B, C; host | 32 a frame |

Measured lane totals, excluding the grid:
- 82.8 MB in f32 with every size up to 2048/1024 built (the 4090); Y alone is 60 MB of that.
- 44.3 MB in f16 with the same sizes (the iGPU).
- 33.0 MB in f32 with 256/32 to 1024/512 (SwiftShader, which builds no 2048).
- 20.8 MB in f32 with 1024/256 alone.

The shared tables are built once, not per lane:
- A: twiddles, the detrend basis, and the rows uniform.
- B: the UV table, 4 B an entry.
- C: MAP, 10 KB a block row a bit map mode, which is 1.6 MB with bit maps 0 and 3 at 128 blocks. Also the lay table
  and the CRC powers in a uniform.

### What is read back

`bh.readBytes` a batch = (4 + 484 recCap) + 2 (4 B blocksMax) + 32 B, rounded up to 8 (and PILOT's 4 B blocksMax +
8 B a frame since the pilots, 2026-09-30 and 2026-10-01). REC goes over whole because
its count exists only on the device. Reading count x 484 instead would take a second round trip.

| build (B = 8) | bytes a batch | a frame |
|---|---|---|
| every size up to 2048/1024 | 504 072 (measured) | 61.5 KB |
| 1024/560 the largest, 2048 left out (the decoder's find mode) | 275 784 (computed from the strides) | 33.7 KB |
| 256/32 to 1024/512 | 252 168 (measured) | 30.8 KB |
| 1024/256 alone | 126 216 (measured) | 15.4 KB |

The front half read back its grid then, 4 MB a frame at 1024; the chain made that copy unnecessary, and the fused
pass 1 (section 12) the grid itself.

### Acceptance

These conditions hold for every result below unless the result says otherwise.
- The frames are the first 16 that the C registers in each of five runs.
- Grids come from the C's own registration and sampler (`_ob_test_sample`).
- Each grid is uploaded as f32 in strip order, at 1024^2 elements a frame.
- Frame and sel are filled as the front half fills them.
- The build is the decoder's: `CURRENT` with the run's own format in its slot. Runs that do not collide share a build.
- The whole chain runs in one command encoder.
- B = 8, INFLIGHT 2, LDPC i16.
- Precision is f16 on the iGPU and f32 on the 4090 and on SwiftShader.
- "The C" means `rxFinish` on the same grids.
- A block counts only if its payload matches the truth stream (`sourceFor(spec.stream)`).

Blocks (GPU / the C). Bad is 0 on every device and run.

| run | format | the C | SwiftShader | 4090 (7 runs) | iGPU (3 runs) |
|---|---|---|---|---|---|
| 12-00-29 | 1024/256 bit map 0 | 288 | 288 | 288 | 288 |
| 21-34-51 | 1024/512 bit map 3 | 568 | 568 | 568 | 568 |
| 20-04-09 | 1024/256 bit map 0 | 389 | 389 | 389 | 390 |
| 02-36-59 | 1024/256 bit map 0 | 32 | 32 | 32 | 32 |
| 06-26-06 | 256/16 bit map 3 | 32 | 32 | 32 | 32 |
| a batch interleaving four 06-26 frames and four 12-00 frames | | 72 | 72 | 72 | 72 |
| simulated 2160 capture | 2048/1024 bit map 3 | 1024 on 8 frames; 256 on 2 | not built | 1024 | 256 (B = 2) |

Per block against the C:
- SwiftShader and the 4090 match the C's verdict on every block.
- The iGPU in f16 matches on all but 3 blocks:
  - 21-34, frame 1: 62 blocks against the C's 63.
  - 21-34, frame 2: 63 against 62.
  - 20-04, frame 0: 15 against 14.
- Iteration counts match the C's on every block tried, with these exceptions:
  - SwiftShader: 510 of 512 on 12-00.
  - 4090: 511 of 512 on 20-04.
  - iGPU f16: 446 of 512, 816 of 1024, 451 of 512 and 323 of 512 on the four 1024 runs, because f16 S moves about
    0.6% of the soft values by one.

Counters and records were right on every frame:
- counters 0 to 2 and 6 equal n, V, npos and subch;
- tried and verified agree with the verdicts;
- there is one record for each verified block, and its iteration count equals ITS.

Every timed submit returned the same blocks as the first run of its batch.

Negative controls. Each device ran each control twice in both 1024 builds, on both lanes, after those lanes had held
decoded batches:
- A noise grid (uniform 0..1) and a zeroed grid were listed and transformed. Their blocks were declined, stalled or
  failed the CRC: 0 blocks.
- A decodable frame with sampled 0, one with valid 0, and one with `sel.x` = 9 were left off every list: no verdict,
  no counter, no record. (`sel` was a vec2u then, x the size; the size is `sel.z` since 2026-09-27.)
- In a build with only 1024/256, the same decodable frame with `sel.x` = 0, 1 or 3 (unbuilt), 4 or 1000 was also left
  off every list.
- A control frame in each of these batches gave the C's count.

GPU ms a frame, from timestamps. Each run made 10 timed submits after 3 warm ones. Each cell gives the median of the
runs' medians, with the minimum over all submits in brackets.
- The six stages were timed as six passes, each with its own timestamp pair.
- "chain" is the sum of those six stages.
- "one pass" is the whole chain in one compute pass.
- "busy" is 16 chains in one submit, the last 12 counted: the device kept busy. In the other two, the device idles
  between submits while the next grids arrive over HTTP.
- Gate and reduce are at most 0.007 ms everywhere and are left out.

| iGPU, f16 (3 runs) | pass 1 | pass 2 | soft | LDPC | chain | one pass | busy |
|---|---|---|---|---|---|---|---|
| 12-00-29, 1024/256 | 0.59 | 0.07 | 0.07 | 4.57 | 5.29 (5.23) | 5.30 (5.21) | 5.17 |
| 21-34-51, 1024/512 | 0.90 | 0.10 | 0.17 | 6.85 | 8.03 (7.88) | 8.00 (7.86) | 7.77 |
| 20-04-09, 1024/256 | 0.58 | 0.06 | 0.06 | 3.02 | 3.71 (3.69) | 3.72 (3.67) | 3.61 |
| 02-36-59, 1024/256 | 0.62 | 0.07 | 0.06 | 6.17 | 6.93 (6.88) | 6.91 (6.84) | 6.77 |
| 06-26-06, 256/16 | 0.025 | 0.005 | 0.005 | 0.079 | 0.13 (0.11) | 0.16 (0.11) | 0.11 |
| simulated 2160, 2048/1024, B = 2 | 3.88 | 0.28 | 0.30 | 1.84 | 6.31 (6.05) | 6.42 (5.73) | 5.51 |

On the iGPU:
- The LDPC takes 81 to 89% of the chain at 1024, and pass 1 most of the rest.
- At 2048 the transform takes 66%.
- Run to run, the medians at 1024 agree within 1.2%.
- At 1024 the chain costs the same, within 1%, as one pass as it does as six.
- The chain misses the 4 ms budget of section 10 on two of the three 1024/256 runs.
- On 02-36 and 21-34 it is about 1.2 and 1.1 times the wasm back half on one desktop thread (5.6 and 7.3 ms, section
  10's figures, measured on 60 frames against these 16).

The 4090's times are not reported: they swung between runs (two states 6.5 times apart on the simulated 2048), as
the standalone LDPC's did (section C).

Section 10's "all" row is not measured here: all 19 runs behind the GPU front half, against the 92 247 blocks. It is
the wiring's acceptance.

## 12. As built: the fused sampler (2026-09-24)

With the GPU back half, pass 1 samples each picture itself and F9 does not run: the grid (4 MB a frame at n = 1024,
f32) is never written or read, and the decoder does not make it. Section 3's plan, built; nothing after pass 1's
load changed. Since 2026-10-22 the fused pass 1 is the only one: the grid variant, `fuse`, `BACK=both` and the
comparison's tests (`test_fuse`, `fuse_page`, archived) are gone, and F9 runs only for the wasm back half. What
follows is the record of the comparison.

- **Two pass 1 variants** (`wgsl/back_transform.mjs pass1Source` `picture: "grid" | "fused"`). The grid one is as
  before. The fused one computes each sample as F9 does: module (g0 + x step, g0 + y step) through `toImage`
  (`common.mjs MAP`), plus the Coons patch of F7's residuals, read by `bilin0` (pixel centres at i + 0.5, clamped to
  the frame's own w, h), in the same order a thread loads its 8 rows of a column quad. The patch is `sample.mjs COONS`,
  which F9 now includes too, so there is one copy of it. A side's curve depends on one coordinate only, so the fused
  pass takes the top and bottom curves and the weight across once a column and the left and right curves and the
  weight down once a row: 24 curves a thread for its 32 samples, where F9 evaluates 128. The sums, the means,
  the FFT and the Y store are the grid variant's code.
- **Bindings**: 7 storage buffers (LISTS, twiddles, PART, Y, BCOUNTS, maps, resid; Y0 until 2026-09-25, when the
  column mean moved into PART's w so the cancel variant could bind REF, section 13), the frame texture, and two
  uniforms: the params and PIC (a size's g0, step, nodes a side and up to 36 lattice coordinates, from the front
  half's tables: `gpu/tables.mjs` g0, step, nodeX, the numbers F9 samples by). Section A's "8 storage buffers plus a
  texture" miscounted: `frames` would be a ninth. The gate now writes every frame's (w, h) after the four lists
  (LISTS is 4 (1 + B) + 2 B words, `listsWords`; 3 B since 2026-09-26, each frame's block count after them,
  `listsBlocks`, and 4 B the same day, each frame's row count after those, `listsRows`), and the fused pass reads the
  edge clamp from there. Workgroup
  memory is the grid variant's.
- **The sampled flag keeps its meaning.** `sel[f]` is written by PICK (F6) in find mode, cleared at the head of every
  find batch, and by the host when it hands the maps over. F9 samples a frame when `frames[f].valid` is set and `sel[f].y`
  >= 2; the gate lists it under the same condition plus a built `sel[f].z`. So the fused pass samples exactly the
  frames F9 sampled and the gate listed. The front half's sample counter (`COUNT.samples`) stays 0 in a fused batch;
  BCOUNTS column 0 (columns transformed, n a listed frame) is the check that every column was sampled.
- **The detrend needed nothing.** The partials come from pass 1's load as before, now of the fused samples; reduce
  and pass 2's spectral subtraction read PART and COEF as before. PART, Y0 and COEF are bit-equal between the
  variants wherever the samples are (below).
- **Wiring** (`decoder.mjs`, `back/back.mjs`, `back/transform.mjs`). `BackHalf.build(..., { picture })` builds the fused
  pass 1 beside the grid one; `bh.bindPicture(ln, { texView, mapsBuf, residBuf })` binds it (the decoder rebinds
  when the texture grows); `bh.encode(target, ln, { fused })` picks it. `FrontHalf.create` `fuse` (default true, with
  `back: "gpu"`). A batch without grids runs no F9 and the lane makes no grid buffer; `gridStride` stays the picture
  capacity (the finder's pick bound, the back half's params). A batch that asks for grids (`BACK=both`) makes the
  grid on first use and runs F9 and the grid pass 1, so the wasm and the GPU back half read one grid. `back: "wasm"`
  is unchanged. `scripts/gpu/back/test_back.mjs` keeps the grid pass 1 (grids from the C's registration).

**The spectrum, fused against grid** (`back/test_fuse.mjs`, page `scripts/gpu/back/fuse_page.mjs`: the whole decoder in find mode on
recorded frames, and on one batch's state the back half run with each variant; the first 16 frames of each run, B 8):

| device | S entries bit-equal | worst entry error / rms | rms error | Y0, PART bit-equal | verdicts equal | blocks grid / fused |
|---|---|---|---|---|---|---|
| iGPU f16, 20-04 / 19-48 / 21-34 / 06-26 | 100% on all four | 0 | exact | 100% | 2080 / 2080 | 413/413, 351/351, 565/565, 32/32 |
| SwiftShader f32, 20-04 (8 frames) | 100% | 0 | exact | 100% | 256 / 256 | 190 / 190 |
| 4090 f32, the same four | 0.18 to 1.2% | 1.4e-4 at most | -92.6 dB worst frame, -94.5 to -109.5 median | 24%, 31% | 2079 / 2080 | 412/412, 349/348, 565/565, 32/32 |

On the iGPU and SwiftShader the fused samples are F9's to the bit, so the arithmetic is the same; on the 4090 the
two shaders' f32 arithmetic is rounded differently by its compiler (which operation was not isolated; the map's
products shared between a column's samples, or contracted differently, would do it). -93 dB sits 30 dB over the f32 transform's own floor (-129 dB against f64, section A) and 33 dB under f16's; one
verdict in 2,080 moved (19-48-40 frame 2, 10 blocks to 9).

**Acceptance** (`scripts/exp/gpu_captures.mjs FINDER=cnn`, defaults, the tree of 2026-09-24 14:10 with only this change):
- 19 x 40 set (753 frames), 4090 f32: 11,425 blocks against 11,425 before, bad 0. 17 frames moved by one or two
  blocks, 9 up and 8 down: 19-48-10 frames 20 (17 to 16) and 30 (10 to 9); 19-48-40 2 (10 to 9); 20-04-09 16 (21 to
  22), 21 (29 to 28), 39 (30 to 31); 20-24-58 9 (9 to 10); 20-28-15 13 (4 to 5), 26 (19 to 20), 28 (20 to 19);
  22-28-38 5 (4 to 2), 22 (14 to 15), 29 (22 to 23), 32 (6 to 7), 36 (22 to 21), 37 (23 to 24); 21-34-51 28 (63 to
  62). On every one the finder found the frame with the same score as before (no frame of the 753 changed its found
  flag or its score), and `BACK=both` on the same tree, which runs F9 and the grid pass 1, gives the old count on
  all 753 frames (the wasm on those grids 11,425 too). So each move is the fused pass's f32 rounding on the 4090
  flipping a marginal block, the size of move the table above predicts.
- 19 x 40, iGPU f16: 11,435 against 11,435, bad 0.
- `scripts/exp/gpu_front.mjs FINDER=cnn` (6 frames a cell, wasm back half, so only F9's regrouping reaches it): every cell
  the same as the tree before, but 45 deg at 63 blocks against 62 (the morning's run of the tree before gave 63:
  the front half's own replay spread); both no-symbol controls 0; 46 of 46 empty slots gave nothing.
- SwiftShader, `RUNS=20-04-09 FRAMES=8`: builds and runs; with `FINDER=cnn` 190 blocks before and after (the C 184),
  bad 0. With the default hand finder it finds no frame, before as after.
- Every frame of every run (5,942 frames), 4090 f32: 93,135 against 93,124 before (+11, 0.012%), bad 0 on all 19
  runs; 152 frames moved by one or two blocks (80 up, 72 down) and none changed its found flag or finder score.

**Time.** Stage timestamps (`PROFILE=1`), before on the tree as it was and after on the same tree with this change, three
repeats each; the iGPU's repeats agree to 0.01 ms. Blocks are the same runs'.

| | sample + pass 1 before | fused pass 1 | whole decode, ms a frame | frames a second (steady) | blocks before / after |
|---|---|---|---|---|---|
| iGPU f16, 1080 run 20-04-09, 48 frames, B 8 | 0.89 + 0.48 | 0.77 | 10.63 to 10.01 | 91 (94) to 97 (100) | 1,341 / 1,341 |
| iGPU f16, 2160 run 19-48-40, 16 frames, B 2 | 0.89 + 0.48 | 0.81 | 25.83 to 25.27 | 37 (38) to 38 (39) | 351 / 351 |

On the 4090 in f32 the same runs gave 1,336 blocks before and 1,337 after on the 1080 run, 349 and 348 on the 2160
run; its times are not reported (`gpu/README.md`).

The same-state test (`scripts/gpu/back/test_fuse.mjs`, the first 16 frames, B 8, 20 submits after 3, alternating, the median):

| | F9 + grid pass 1 | fused pass 1 | saved |
|---|---|---|---|
| iGPU f16: 20-04-09 (1024/256) / 19-48-40 (1024/256, 2160 frames) / 21-34-51 (1024/512) / 06-26-06 (256/16) | 1.347 / 1.344 / 1.576 / 0.134 | 0.764 / 0.805 / 0.905 / 0.056 | 43 / 40 / 43 / 58% |

Where the fused pass's time goes: `scripts/gpu/back/test_fuse.mjs PROBES=1` times the pass again with one part cut out, each probe
alternated with the whole pass on the same state (timing only, never a decode; 8 frames, B 8, 20 submits, median).

| ms a frame | whole | no radial iteration | no texture reads | no sampling (loads, sums, FFT, Y store) |
|---|---|---|---|---|
| iGPU f16, 20-04-09 / 19-48-40 | 0.763 / 0.806 | 0.533 / 0.572 | 0.588 / 0.592 | 0.206 / 0.208 |

On the iGPU the sampling is about 0.56 ms of the pass's 0.76, and the map's radial iteration (six steps a sample,
`common.mjs MAP`) about 0.23 of that, the four texture reads about 0.18. The parts overlap, so they do not add.

**Memory a lane** (the decoder's, B 8, 1080 frames, 1024 picture): a lane 69 MB to 37 on the iGPU in f16 and 78 to 46 on the 4090 in f32 (the
grid, 4 x 1024^2 x 8 bytes, is 32 MiB); B 2 on the 2160 run, 35 to 27 on the iGPU. The 19 x 40 set's largest lane,
165 MB to 133 on the 4090 (334 to 270 over its two lanes) and 149 to 117 on the iGPU (302 to 238). Per frame, the
grid's 4 MB written by F9 and read by pass 1 are gone at n = 1024 (16 MB at 2048).

Caveats: the 4090's replay moves marginal blocks both ways as any f32 change there does; the iGPU's samples are
bit-equal, so its blocks move only with the front half's own replay spread. The numbers are this desktop's two GPUs;
no phone has run it. 

## 13. Straddle cancellation (2026-09-25, deleted 2026-10-22)
Pass two of a batch: a capture that straddles a picture change mixes two painted frames under one pose, and the
receiver holds most of both as verified blocks, so it painted them on the device, fitted the mixture's two shares as
planes over three bases each, subtracted them and read the frame again, in the batch's own command buffer after pass
one. Off by default from late 2026-09-25; from 2026-10-07 it no longer built, since its paint and gate painted one code
rate and the format's rate profile has four. Deleted on 2026-10-22: the camera's phase, held by the pilots (SPEC 7.3),
keeps captures from straddling at the source.
- Gained (4090 f32, every frame, bad 0, the one-rate format): 09-14-21 (60 painted at 60 captured) 6,156 to 8,575
  blocks, 02-36-59 2,830 to 4,195; gains only on frames it cancelled.
- Cost (the iGPU, f16): 3.3 ms a frame on a stream that straddles everywhere (10.83 against 7.57), nothing on a clean
  one once frames under half their batch's best were the only ones read again.
- Its code (the gate, the paint, the bases and fit, pass 1's cancel variant and the samples it kept, pass two's LDPC
  stop rule `funnel2`, the host, the receiver's `?gcancel=on`) is in the repository's history before that date. The
  sender kept its two inverse transforms (`wgsl/inverse.mjs`, irows and ipic) and the clip (`paint.mjs`).

## 14. The shift stage (2026-09-27, deleted 2026-09-28)
Pass one's reread through each frame's own sub-sample shift: the C's dd mode 2 on the device (the verified blocks'
codewords as pilots, a shift fitted a frame, the failed blocks read again through it past 0.13 samples), three
dispatches after pass one's LDPC. Deleted on 2026-09-28 by the rule against patch filters (no filter that patches a
symptom; a simple GPU pipeline), with the C's shift pass. The leftover shift it corrected (0.16 samples median in
the GPU's grids of 07-56-38, a +y bias the C's grids share at 0.10) is F7's registration to fix at its source.
- Gained (4090 f32, every frame, `research/results/prune/before/gpu_captures.txt`): +1,637 blocks on 07-56-38 (20,139
  verified with it) and +3,659 on 17-59-29 (23,485).
- Cost: the reread (`reldpc`) 1.4 to 4.7 ms a frame live, median 3.4 over 49 stats rows (S26 Ultra, two LIZARD-512
  at 2:1, the stage timing menu on, 2026-09-28; STATUS "The 2:1 capture along the frame's long side"), against under
  0.12 in the bench.
- Its code: `archive/shift-pass/gpu/` (`back/shift.mjs`, `wgsl/back_shift.mjs`, as they stood) and
  `archive/shift-pass/removal.patch` (every hunk of the deletion, its README the restore). Its variants
  in the soft stage (`shift`) and the LDPC (`read: "reread"`, and the first read's hard decisions in L past the soft
  values) went with it; the tree before the deletion is `archive/before-copies/prune/before/`. PERMW, which it had taken from
  the cancel stage, went with that stage (13).
