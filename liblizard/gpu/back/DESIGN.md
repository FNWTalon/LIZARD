# The GPU back half: design (2026-09-24)

Input: the n x n sampled picture the front half writes (F9, `wgsl/sample.mjs`), one a frame, B frames a batch.
Output: CRC-verified blocks with their ids, and nothing else, read back once a batch. The reference for every rule
is `src/focus.c` `focus_finish_bits` (lines 955 to 1226), `src/ldpc.c` `ldpc_decode_stall`, `src/layout.c` `ob_crc32`.
The decoder is judged on blocks that pass their CRC-32 against the truth stream, on ms a frame at batch B on both
GPUs, and on a negative control; never on parity with the C. Every table named below was verified today by calling
the export from node (`init()` from `sim/ob.mjs`); the checks and their counts are in sections 8 and 9.

Formats in play: n / subch = 256/32, 512/128, 1024/560, 2048/1024 (the top of each picture size, `gpu/tables.mjs`)
and the recorded runs' 256/16, 512/64, 1024/192, 1024/256, 1024/512. Every format is ONE LDPC code (rate 3/4:
n 5088, k 3816, m 1272, z 106, 12 block rows of 15 data slots) and one block size, 473 payload bytes, the first 4
the block id (LE). Blocks a frame = subch / 8. Tiers exist in the C (`focus_init_tiers`) and not in the format
(`FOCUS_RATE`), so the design carries one code and one tier; every table is read per tier anyway.

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
  nothing and is not stored. This is the C's spectral subtraction (focus.c lines 1001 to 1010) to the letter; a
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
  to 5119 of a block carry whitened zeros and are not read. k = 3816 = 473 * 8 + 32 exactly, so the C's "known
  zeros past the payload" loop (focus.c line 1110) is empty at this rate.
- **C LDPC**: normalised min-sum, layered by block row, norm 13/16, posteriors clamped to +-8191, messages to
  +-127, cap 30 iterations, sweep direction alternating each iteration, the stall rule (violated checks after
  iteration 9 above 0.95 of the count after iteration 1 gives up, `FOCUS_STALL_IT`, `FOCUS_STALL_RATIO`), exit on a
  clear syndrome. Integer arithmetic, kept integer: the norm and caps were set on 78 000 blocks
  (`test/ldpc_ablate.c`, focus.c line 1097) and mean what they were measured to mean only in integers. Given
  equal LLRs the decisions and the iteration count equal the C's bit for bit under the stop rule `"c"`; the first
  read's default since 2026-09-28 is the learned rule (section 6; the funnel from 2026-09-24), which gives up on a
  failing codeword by a small net's reading and verifies the same codewords but for its budget of 1 in 1,000.
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
| LISTS, ARGS | gate -> all | u32 [SLOTS sizes][1 + B] (SLOTS = 6 since 2026-09-27, `wgsl/common.mjs`): count, then frame indices, then each frame's (w, h) and each frame's block count (2026-09-26: the version F8 read in sel.y = 1 + v (or the held one, 2026-09-27), up to the size's blocks, else the size's; soft and LDPC run a frame's first count blocks), then each frame's row count (2026-09-26, `listsRows`: the disc rows that count takes, section 3 "Rows a frame"); u32 [SLOTS][12] indirect args, (x, y, z) for pass 1, reduce, pass 2 (the most rows a listed frame of the size takes) and the blocks slot (blocks, 1, count) | 376 B + 288 B at B = 8 (`listsWords`, `argsWords`) | | | |
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
storage bound; measure both. Four pipelines a pass (one per n), dispatched indirectly from ARGS over LISTS; a
workgroup for a frame of another size returns before its first barrier.

Rows a frame (2026-09-26). Every size is built at its top (`gpu/tables.mjs` CURRENT) and a frame decodes the
version its word names, a prefix of the top's coefficients, so it needs only the rows its blocks' coefficients
reach. `transform.mjs rowsByCount` gives rowsBy[c], the rows the first c blocks take (block b is pos[2560 b ..
2560 b + 2559], sub-channels 8b to 8b + 7), from the top's own pos table; `scripts/gpu/back/check_rows.mjs` holds every version
of every size to its own disc's V (128 of 128: 256 41 to 81, 512 91 to 167, 1024 172 to 341, 2048 343 to 457). The
gate's uniform carries it as rows[c][size]; the gate writes each frame's rows beside its block count in LISTS
(`listsRows`), pass 1 stores Y rows 1 to rows - 1 (the FFT itself is unchanged: every column still runs whole), pass 2
returns on a row past the frame's before its first barrier (LISTS is read-only there, so the branch is uniform) and
its indirect dispatch takes the most rows any listed frame of the size needs. S past a frame's rows is left stale:
the soft stage reads only its first count blocks' coefficients, all inside them. gate2 copies the rows into LISTS2
and sizes pass 2b the same way. The rule since 2026-09-26 (every frame decodes on its own, no memory across
frames or batches) makes the word the only source: a frame whose word is not read runs its size's top, rows and
blocks, so the recordings older than 2026-09-24's word (every timing run of the protocol) keep the top's cost.
irows and ipic (the cancel stage's paint) still run the size's V: a reference's rows past its version are zero.

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
(`focus_rx_ldpc`) and C is tested on the C's own terms (`focus_rx_bits`).

## 5. The decline gate

BLK.declined a block from blk_est < 0.15. C's workgroup for a declined block writes verdict 2, ITS -2, and returns
before touching L: a workgroup with no work, which the shape rules allow. The gate and the stall rule are the two
thresholds this decoder keeps from the C. They are format decoder rules measured on 78 000 blocks, and on the
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
the C paints on every slot (section 9), so the H the implementer reconstructs is the C's.

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
  `stop/` (`scripts/gpu/back/stop/README.md`).
- `"c"`, the C's rule: the stall test after iteration 9 and the cap of 30. The check that the kernel's min-sum is the
  C's: under it `test_ldpc` holds the GPU to `ref_ldpc.mjs` and that to the C (`rxLdpc`) on every block. Its shader is
  the one before 2026-09-24, byte for byte, in all six variants.
- Pass two keeps its own rule, `"funnel2"` (13.6): the stall test and six checkpoints, out of this change's scope.

The learned rule. The question: given what lane 0 can hold after an iteration, is the codeword hopeless? It is
answered on neutral codewords only (`scripts/gpu/back/stop/neutral.py`: the format's code through a generic channel and the soft
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

Where the iterations went, from `scripts/exp/ldpc_stop.mjs`. The recorded frames went through the GPU decoder on the 4090
(FINDER=cnn, the defaults) with the grids read back. Each grid went through the wasm's soft values and a JS copy
of this decoder (`ref_ldpc.mjs makeTracer`), run to the cap with no stall rule, which keeps each codeword's count
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
configuration at a time. Verified today at every format above (shapes printed from the exports).

| table | export (call) | shape |
|---|---|---|
| n, bw, vblocks, subch, blocks, npos | `M._focus_shape(p)`, `Focus.shape()` | 6 i32 |
| pos, decoded to (u, v) | `M._focus_pos_ptr()`, `Focus.posTable()`; v = (p div (n bw)) bw + (p mod (n bw)) mod bw, uw = (p mod (n bw)) div bw, u = uw < n / 2 ? uw : uw - n | subch * 320 u32; entry 0 is (1, 0) |
| cx, qx, X1, X2, sx2, sq2 | `M._focus_tables_out(p)`, `Focus.tables()` | 6 n + 2 f32 (1538, 3074, 6146, 12290) |
| block first sub-channel, count | `M._focus_block_subs(pFirst, pCount)`, `Focus.ldpcCode().first, .count` | blocks i32 each; first = 8 b, count 8 |
| block tier | `M._focus_block_tiers(p)` | blocks i32, all 0 |
| LDPC dims and schedule | `M._focus_ldpc_tables(tier, pDims, pLay)` (pLay = 0 sizes it), `Focus.ldpcCode().code` | dims: n 5088, k 3816, m 1272, z 106, zp 112, mb 12, kb 36, norm 13, slotsMax 17, slotsTotal 204, slots 180; lay 373 i32 |
| block bytes | the return of `M._focus_setup(...)`, `Focus.blockBytes` | 473 |
| bit map mode | `M._focus_bitmap_get()` | 3; 0 for a replay of a recording without `bitmap` (`recordedSpec`) |
| perm, inv, white, MAP | not exported; regenerated in JS from section 1 (`bitmap.mjs`) and verified against what the C paints (section 9). C to add: `int focus_bitmap_tables_out(int tier, uint8_t *white, int32_t *perm)` in src/wasm.c copying `F.ws->white` (subch * 640 B) and `F.ws->perm[tier]` (`code[tier].n` i32) | perm 5088; white subch * 640 |
| g0, step | `M._focus_grid_out(p)` | 2 f32 |
| CRC-32 | `M._ob_test_crc_hash(p, len)`; `M._ob_test_crc_pack(bits, 473, dst, out2)` | |
| decline bar, verdicts | `M._ob_test_crc_gate_out(est, blkEst, bar, declined)` | blocks each |
| truth stream | `streamBlock(id, len)` (sim/ob.mjs, `_stream_fill`); `sourceFor(spec.stream)` (sim/phy.mjs) | |
| the C's spectrum | `M._focus_spectrum_out(re, im)` after a finish, vblocks * n * 64 f32 each in the C's layout; `Focus.measure(true)` then `read()`: 2 * 320 * subch f32, the detrended coefficients in pos order | |
| the C's verdicts | `Focus.rxFinish().ok`; `Focus.blockStats()` its (-2 declined, -3 stall, -1 cap, > 0 iterations) and est | blocks |
| a grid from the C's registration | `M._ob_test_sample(img, iw, ih, 1, 2, dims, state, grid)` | n x n f32 row major, pitch n |
| finish from soft values, decisions, verdicts | `Focus.llrView()` + `rxLdpc(est)`; `rxBits(est, bits, its, n)`; `rxAssemble(bytes, stride, verdicts, its, est, quad)` | |
| the exact symbols sent | `Focus.measure(true)`, `encode(blocks)`, `sent()` | 640 subch int8, +1 / -1 |

Missing exports: `focus_bitmap_tables_out` (recommended: makes the regeneration checkable without an encode).
Optional: `int focus_llr_out(int8_t *llr, float *est)`, the C's own slot order LLRs after a finish (focus.c reuses
one block's worth at a time, so it would have to keep subch * 640 bytes), and `focus_ldpc_encode_out(tier, data,
cw)` for synthetic curves. Neither is needed: the JS reference in section 9 stands in for the first and the JS
encoder for the second, both checked against the C.

## 9. Standalone tests against the wasm, no GPU front half

Shared loader `scripts/gpu/back/frames.mjs`: for a run under `research/captures` and a frame list, read `meta.json` and the
`.gray` frames (`meta.sizes[i]` or `meta.w, meta.h`), `spec = recordedSpec(meta.config.spec)`, `phy = await
makePhy(spec)`, `fc = new Focus(spec.n, spec.subch, 1, { span: spec.span ?? 0, bitmap: spec.bitmap })`.
Per frame: (a) the grid: `_ob_test_sample` (the C's own registration and sampler) reordered to strips,
strip[(x div 64) n 64 + y 64 + x mod 64] = grid[y n + x]; (b) the C's finish on that grid: `rxAcquireQuadNoImage(64,
64, [8, 8, 56, 8, 56, 56, 8, 56], 0, 1, { mesh: 0 })` as `harness/score.mjs` does, `measure(true)`, `gridView(n).set(strip)`,
`d = rxFinish()`, then `read()`, `_focus_spectrum_out`, `blockStats()`; (c) `phy.tally({ ...d, found: 1 }, 0, 0)`
for seen and bad against the truth stream. Checked today: this path decodes the same blocks as the one call
decode on 24 of 24 frames (12 each of 02-36-59 and 21-34-51), and `read()` equals `_focus_spectrum_out` at every
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

## 10. Three implementers

Files: host `gpu/back/<stage>.mjs`, shaders `gpu/wgsl/back_<stage>.mjs`, test `gpu/back/test_<stage>.mjs`. Each
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
  ascending n (up to 4; a size's index is its position, and `sel[f].z` indexes it). `t.lane({ gridBuf, gridStride,
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
  sel[f].y >= 2 and sel[f].z names a built size below 4; it used to fold an index of 4 or more into size 3.
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
  build it, so the test simulates 2048 on real adapters only).
- Measured (`scripts/gpu/back/test_transform.mjs`, 16 frames a run registered by the C's own sampler, B = 8, 5 submits a batch,
  the median; blocks = the disc through the soft-value reference and the C's LDPC, the C's own count beside):

  | device, precision | 1024/256 (20-04-09) | 1024/512 (21-34-51) | 256/16 (06-26-06) | 2048/1024 (simulated 2160) |
  |---|---|---|---|---|
  | iGPU f16, ms a frame (gate + pass 1 + reduce + pass 2) | 0.60 (0.001, 0.538, 0.001, 0.060) | 0.85 (0.001, 0.762, 0.001, 0.090) | 0.075 | 3.64 at B = 2 (3.37 pass 1) |
  | iGPU f16, S vs f64 dB (worst frame) | -60.0 (-58.1) | -59.4 (-58.4) | -63.2 (-61.6) | -59.9 (-55.9) |
  | iGPU f16, blocks GPU / C, bad | 390 / 389, 0 | 568 / 568, 0 | 32 / 32, 0 | 256 / 256, 0 |
  | 4090 f32, ms a frame | 0.014 (0.001, 0.009, 0.002, 0.003) | 0.014 | 0.006 | 0.051 at B = 8 |
  | 4090 f32, S vs f64 dB | -129.5 | -129.0 | -131.9 | -128.7 (1024 / 1024 blocks) |
  | iGPU f32, ms a frame | 0.96 | | 0.088 | |
  | budget (section 10) | 1.5 / 0.10 | 2.5 / 0.15 | | 6 / 0.4 |

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
  compute pass, timestamps, median of 10 repeats: 4090 f32 0.002 at 1024/256, 0.003 to 0.009 at 1024/512; iGPU
  f32 0.06 / 0.13, f16 0.05 / 0.11 (the budget was 0.3 at 1024/512). Indirect dispatch from an ARGS slot changes
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
  storage, which the uniformity analysis accepts as uniform at a uniform index). No ARGS slot is read.
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

| variant | iGPU (RDNA-2, 2 CUs) ms a frame | us an iteration | 4090 ms a frame | 4090 ms a batch |
|---|---|---|---|---|
| base | 11.0 / 7.0 / 13.0 | 21 | 0.26 / 0.27 / 0.26 | 2.04 / 2.16 / 2.07 |
| hoist | 7.2 / 4.6 / 9.4 | 14.1 | 0.16 / 0.16 / 0.16 | 1.28 / 1.28 / 1.28 |
| reg | 7.2 / 4.6 / 9.3 | 13.7 | 0.25 / 0.16 / 0.16 | 2.01 (min 1.13) / 1.28 / 1.28 |
| i16 (default) | 6.3 / 4.0 / 8.2 (7.5 over all 56) | 12.1 | 0.22 / 0.22 / 0.26 | 1.79 / 1.79 / 2.08 |
| flags | 5.6 / 3.6 / 7.5 | 11.0 | 0.23 / 0.12 / 0.12 | 1.84 / 0.97 / 0.97 |
| i16flags | 4.6 / 3.0 / 6.2 | 9.0 | 0.43 (min 0.24) / 0.17 / 0.20 | 3.44 / 1.39 / 1.62 |

Scaling with codewords in flight. iGPU, i16, 02-36: 32 codewords 6.97 ms (477 iterations, 14.6 us each), 64
6.99 (13.6), 128 20.4 (12.6), 256 50.8 (12.1); 21-34 at B = 32, i16: 16 frames in flight (1024 codewords, 10 442
iterations) 7.87 ms a frame, 12.1 us an iteration, and 32 in flight (2048 codewords) 8.03 ms a frame, 11.9 us, blocks
equal to the C's (`IGPU=1 B=32 RUNS=21-34:32 VARIANT=i16 REPEATS=3`). The 13.7 ms a frame this line gave before was that point measured
with VARIANT=base (21 us an iteration), not i16. Time is proportional to iterations run from 64 codewords up: nothing is left to hide, the WGP is
instruction bound (six resident codewords in place of four, i16 against reg, bought 1.13x; the flag gate,
which removes a third of the reads and the index arithmetic, 1.27x). 4090, base, 21-34: 64 / 128 / 256 / 512
codewords 2.06 / 2.06 / 2.07 / 2.07 ms a batch, 1024 2.86, 2048 4.78: flat to about a thousand codewords, a
latency floor of about 5 us a row step (3.3 under reg), so a bigger batch is free there. Budgets: the 4090 is
under its 0.3 ms a frame with every variant; the iGPU misses 3 ms on 02-36 and 1.5 on 12-00 by 2.1x and 2.7x
with the default, 1.5x and 2x with i16flags, and sits beside this desktop's wasm (4.13 / 5.43 ms on 02-36 /
21-34). Caveats: the 4090's LDPC times are bimodal between runs of the same test on the same batch, by 5x to 12x:
i16 at 12-00 read 9.86 ms a batch in one run and 1.79 in two others, 02-36 21.0 against 1.77 to 2.21. Read every
4090 time in this section as approximate and the minimum beside the median. What is known: the 4090 idles at 210
MHz between the test's jobs (the wasm side takes about a second a batch) and a job of a few ms does not always
raise it; section 11 gives the chain's times as the median and min over seven runs. The iGPU shares package power
with a CPU other jobs were loading (load average 10 to 28 during these runs); an iGPU submit of about 2 s or
more (base at B = 32, 5 timed passes: 3 s) comes back empty with no error. Not built: the two codewords a workgroup variant, and R packed at 36 bits a check (9 words for
8 checks), which with i16 posteriors would be 15.9 KB, two codewords in a 32 KB phone core.

## 11. As composed (2026-09-24)

`back.mjs` is the chain behind the front half. `BackHalf` owns A, B and C, their tables, and one set of per-frame
strides from `dims.mjs derived(sizes, { B, cap })`, which `soft.mjs` and `ldpc.mjs` also size themselves from. B
and C each worked out their own strides before this, from their own size lists, and could disagree on where frame
f's slots sit. `scripts/gpu/back/test_back.mjs` runs the chain on the device as the decoder will.

### What the wiring calls

```
const bh = await BackHalf.build(device, { B, sizes, precision, cap, variant, log });
const ln = bh.lane({ gridBuf, gridStride, framesBuf, selBuf });   // a front-half lane's own buffers
bh.encode(enc, ln, { ts: { qs, base } });   // six compute passes, a timestamp pair each at qs[base + 2k]
bh.encode(pass, ln);                        // or the six dispatches appended to a pass the caller holds open
const used = bh.readback(enc, ln, readBuf, off);   // REC, V, ITS, BCOUNTS copied to readBuf at off; used = bh.readBytes
const r = parseReadback(mappedArrayBuffer, off, bh.dims);
//   r.records: [{ frame, block, its, id, payload (473 B, id first) }], the verified blocks only
//   r.frames[f]: { verdict, its (blocksMax each), counts (COUNT: columns, rows, stored, tried, verified, records,
//   subch, iterations) }
bh.destroyLane(ln);
```

- `sizes`: the decoder's four, at the size index `sel[f].z`, each `{ n, subch, span, bitmap }` or null. If `bitmap`
  is left out, the format's own is used (3, `focus_setup`'s). A replay must pass `recordedSpec`'s value: 0 for a
  recording made before 2026-09-23. With the wrong bit map, every block fails its CRC and nothing else goes wrong.
- Pass null for a size whose picture does not fit the lane's grid (n^2 > gridStride). The finder cannot pick such a
  size, and leaving it out sets blocksMax, and with it every stride and the readback, by the largest size that can
  occur. With 2048/1024 built, blocksMax is 128. With 1024/560 as the largest, it is 70.
- A size whose pass 1 needs more workgroup memory than the device offers is left unbuilt, and the build logs it
  (2048 in f32 on SwiftShader's 32 KB). The gate leaves a frame off every list when:
  - its size is not built;
  - its `sel[f].z` is 4 or more;
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
- Device limits: workgroup memory at the adapter's maximum. The LDPC needs 20 848 B. Pass 1 needs 14 KB (f16) or
  28 KB (f32) at 1024, and 28 KB or 45 KB at 2048. Each stage binds at most 8 storage buffers. `FrontHalf.create`
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

`bh.readBytes` a batch = (4 + 484 recCap) + 2 (4 B blocksMax) + 32 B, rounded up to 8. REC goes over whole because
its count exists only on the device. Reading count x 484 instead would take a second round trip.

| build (B = 8) | bytes a batch | a frame |
|---|---|---|
| every size up to 2048/1024 | 504 072 (measured) | 61.5 KB |
| 1024/560 the largest, 2048 left out (the decoder's find mode) | 275 784 (computed from the strides) | 33.7 KB |
| 256/32 to 1024/512 | 252 168 (measured) | 30.8 KB |
| 1024/256 alone | 126 216 (measured) | 15.4 KB |

The front half reads back its grid today, 4 MB a frame at 1024. The chain makes that copy unnecessary.

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

| 4090, f32 (7 runs; busy on 3) | pass 1 | pass 2 | soft | LDPC | chain | one pass | busy |
|---|---|---|---|---|---|---|---|
| 12-00-29, 1024/256 | 0.010 | 0.004 | 0.002 | 0.223 (0.185) | 0.245 (0.209) | 0.242 (0.204) | 0.264 (0.213) |
| 21-34-51, 1024/512 | 0.045 (0.012) | 0.019 (0.005) | 0.009 | 0.418 (0.218) | 0.526 (0.247) | 0.317 (0.255) | 0.301 (0.249) |
| 20-04-09, 1024/256 | 0.010 | 0.004 | 0.002 | 0.185 (0.165) | 0.209 (0.190) | 0.205 (0.181) | 0.214 (0.186) |
| 02-36-59, 1024/256 | 0.010 | 0.004 | 0.002 | 0.226 (0.206) | 0.249 (0.231) | 0.246 (0.212) | 0.344 (0.235) |
| 06-26-06, 256/16 | 0.003 | 0.003 | 0.002 | 0.009 | 0.020 (0.019) | 0.018 (0.017) | 0.026 (0.020) |
| simulated 2160, 2048/1024, B = 8 | 0.444 (0.043) | 0.167 (0.012) | 0.016 | 0.092 (0.033) | 0.730 (0.102) | 0.125 (0.105) | 0.126 (0.107) |

On the iGPU:
- The LDPC takes 81 to 89% of the chain at 1024, and pass 1 most of the rest.
- At 2048 the transform takes 66%.
- Run to run, the medians at 1024 agree within 1.2%.
- At 1024 the chain costs the same, within 1%, as one pass as it does as six.
- The chain misses the 4 ms budget of section 10 on two of the three 1024/256 runs.
- On 02-36 and 21-34 it is about 1.2 and 1.1 times the wasm back half on one desktop thread (5.6 and 7.3 ms, section
  10's figures, measured on 60 frames against these 16).

On the 4090, 12-00, 20-04, 02-36 and 06-26 hold within 16% across runs. The other two do not:
- 21-34 drifts down within each run, from 0.5 to 0.75 ms on the first submits to 0.25 to 0.35 on the last. A clock
  rising under load would do this; the clock was not read.
- The simulated 2048 sits in one of two states 6.5 times apart, 0.11 or 0.73 ms a frame. The state switches within a
  run, and the one-pass and busy submits land in either. Most of the difference is in the transform: pass 1 is 0.048
  against 0.44 and pass 2 is 0.013 against 0.17, while the LDPC moves only 2.4 times. The transform passes are
  bandwidth bound and the LDPC is latency bound, so a memory clock left low would fit this, but that is not measured.
- This is the same 5x to 12x swing the standalone LDPC showed (section C). Take the minimum as what the 4090 does
  once it is busy. Either way, it finishes the back half of a 1024 frame in 0.2 to 0.5 ms.

Section 10's "all" row is not measured here: all 19 runs behind the GPU front half, against the 92 247 blocks. It is
the wiring's acceptance.

## 12. As built: the fused sampler (2026-09-24)

With the GPU back half, pass 1 samples each picture itself and F9 does not run: the grid (4 MB a frame at n = 1024,
f32) is never written or read, and the decoder does not make it. Section 3's plan, built; nothing after pass 1's
load changed.

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

## 13. Straddle cancellation: pass two (2026-09-25)

The plan (2026-09-25): a straddled capture mixes two painted frames
under one pose; the receiver already holds most of both (the verified blocks of the neighbouring frames), so it
paints them, fits the mixture's two shares as planes over three bases each, subtracts, and reads the frame again.
Everything runs on the device on pass one's results, with no host round trip, inside the batch's own command buffer
after pass one. From late 2026-09-25 to 2026-09-27
it ran a batch later, in the next batch's command buffer (13.8, reverted). `encode` is pass one, `encodeCancel` pass
two. This section is the format of the buffers and calls the groups code against, and what is built.

### 13.0 Pass one's changes (built, checked)
- The column means ride in PART's w (pass 1 stores `vec4f(S1, SC, SQ, mean)`, pass 2's row v = 0 reads `part[..].w`):
  no Y0 buffer, so pass 1 binds 7 storage buffers and the cancel variant may bind REF as its eighth. Bit-equal by
  construction (the same f32 mean): `test_fuse` S 100% bit-equal, Y0 (PART's w) and PART bit-equal, blocks equal
  on SwiftShader (414, 339, 566, 32) and the iGPU f16 (413, 338, 563, 32); `test_back` 288, 568, 389, 32, 32, the
  C's, bad 0. Y holds V rows a frame (`yStride` = V n): the paint's inverse rows use a frame's Y slot as scratch
  for a whole disc, rows 0 to V - 1.
- The LDPC returns at once on a block whose verdict is already 1 (verified in pass one), read through the
  workgroup flag (`workgroupUniformLoad`: to the uniformity analysis a read_write load is not uniform, so a bare
  `if (V[slotV] == 1u)` does not compile). V is zero in pass one (the gate zeroes it), so pass one is unchanged:
  `test_ldpc`'s table identical before and after.
- The split CRC-32 is one WGSL function, `CRC_SPLIT({ packw, crcw, crcStep, pw })` in `wgsl/back_ldpc.mjs`,
  emitting `fn crcOf(i: u32) -> u32`: every lane calls it with its lane index once the 473 payload bytes are in
  `packw` (120 u32, four bytes a word in memory order, byte 0 of word 118 the last); it holds two barriers and
  returns the CRC to every lane. The LDPC calls it; the paint calls it with its own array names.
- `BUTTERFLIES`, `store`, `firstStage`, `laterStages` and `V8` are exported from `wgsl/back_transform.mjs` for the
  paint's inverse transforms (the same kernels, conj in and conj out).

### 13.1 Buffers (in the BackHalf lane unless shared; `dims.mjs derived()` gives every stride)
nmax the largest built n; cb 4 (f16x2) or 8 (f32x2); R = `refSlots` = min(floor(B / 2) + 1, 16), 4 with a 2048
picture built; NONE = 0xffffffff.

| name | writer -> reader | layout | bytes at 1024/560, B 32 |
|---|---|---|---|
| PLAN | gate2 -> zero, paint, fit, solve, carry | words: KEYTAB [128] { key, bits[4], slot, count, tag }, PTR [128][blocksMax] (record index + 1, 0 none), FR [B] { fb, fa, slotA, slotB, nrefs, short, T, keyOwn }, CARRY_SEL { slot, key, count, size }, SLOTS [R + 1 rounded up to 4] (the KEYTAB entry painted in slot r, NONE unused; SLOTS[R] 1 when the carry slot is a reference), KVER [128] (2026-09-26: a KEYTAB entry's version, the fewest blocks a frame holding one of its records was listed with; ipic clips a slot at that version's level, `paint.mjs` `limTable`, the size's own where NONE); offsets from `planLayout` | 41 KB |
| LISTS2, ARGS2 | gate2 -> pass1c, reduce2, pass2b, soft2, ldpc2, fit, solve | exactly `listsWords(B)` and [SLOTS][ARGS_WORDS] as pass one's, over the short frames with references, by size; the (w, h), block count and row count blocks copied from LISTS; pass 2b's workgroups the most rows a listed frame takes | 0.7 KB |
| CANCEL | gate2 (header) and solve (coef, ok) -> pass1c (uniform), host | [B] { slotA, slotB, nrefs, ok, coef[18] f32, pad[2] } = 24 words; slotB = slotA when nrefs is 1 | 3 KB |
| BCOUNTS2 | gate2 zeroes; pass1c, pass2b, soft2, ldpc2, solve | u32 [B][8], pass one's columns for pass two: 0 columns, 1 rows, 2 stored, 3 the solve's k plus blocks tried, 4 verified in pass two (= gained), 5 records, 6 sub-channels, 7 iterations | 1 KB |
| PIC | ipic -> blur | a slot: the picture row-major at its own n (f16 element y n + x, four adjacent x a vec2u), `picStride` = 2 nmax^2 bytes a slot | 16 x 2 MiB |
| REF | blur, bmeans -> fit, pass1c; the carry in (a copy) | a slot: bases, one 24 B record a (quad of columns, row) at byte ((x / 4) n + y) 24 of the slot, = 3 x vec4<f16>: base0 x0..x3, base1 x0..x3, base2 x0..x3; then at `refMeans` = 6 nmax^2 the means, vec4f (bbar0, bbar1, bbar2, 0) a row; `refStride` = 6 nmax^2 + 16 nmax bytes; R + 1 slots, slot R the carry | 17 x 6.0 MB |
| BPART | blur -> bmeans | [R][3][nmax][nmax / 16] f32 tile row partials | 16 x 0.75 MiB |
| FITP | fit -> solve | [B][nmax / 32][208] f32: A's upper triangle row-major (a <= c, 171) then rhs (18), 19 pad; `fitpStride` = 832 `fitWorkgroups(nmax)` = 832 (nmax / 32) bytes a frame (since the fit takes every fourth row, 2026-09-25) | 0.8 MB |
| CARRY (shared, made with the back half) | carry of batch N -> the carry in of batch N + 1 (a copy into REF slot R on the encoder, before the compute pass) and gate2 (its header as a uniform) | { valid, key, count, size } then padding to 256 B, then one REF slot's bytes | 6.0 MB |
| PERMW (shared) | host -> paint | [blocksMax][5120] u16 a bit map mode in play: codeword bit (0x7fff a zero slot) or white << 15 (`bitmap.mjs permTable`); made and filled by `back/paint.mjs` (2026-09-28; allocated by `Cancel.build` and filled by the paint until 2026-09-27, then the shift stage's, section 14) | 1.3 MB a mode |
| KEEP (the transform's lane, `Transform.lane` with cancel) | pass 1, either picture variant -> pass 1c | [B] the frame's samples as unorm16, quad-major: vec2u (four adjacent x) at f sampStride + (x / 4) n + y, `sampStride` = nmax^2 / 4 vec2u (13.7) | 32 x 2 MB |

Scratch reused: the paint's disc is S's frame slot r, its inverse rows Y's frame slot r (r < R <= B). S's slot r
holds frame r's spectrum from pass one until then: the ZERO shader clears every used slot's disc inside the pass,
right before the paint (a `clearBuffer` before the pass would be overwritten by pass one's pass 2). A short
frame's own S and Y are rewritten by pass two's chain before soft2 reads them.

Memory: +147.1 MB a lane at 1024/560 f16, B 32 (PIC 32.0, REF 102.3, BPART 12.0, FITP 0.8; 149.6 with FITP at 3.3
before the fit took every fourth row), 4.6 MB a frame, and KEEP 2 MB a frame more in the transform's lane (64 MB
at B 32, 13.7); the carry 6.0 MB shared; Y grows 4 KB a frame (f16). At 2048/1024 (R 4) +165.8 MB a lane and KEEP
8 MB a frame (256 MB at B 32), the carry 24 MB. The Batcher prices a frame from a one-frame lane, where R is 1:
+16.8 MB a frame at 1024 (67.1 at 2048; the stage 14.8 and 59.1 of it, KEEP the rest), three times
the B 32 rate, so its Bmax under a tight budget is smaller than the lanes would need to be (B_CEIL, not memory,
binds on the S26 and the 4090; on SwiftShader's 1 GiB the smoke planned lanes of 22).

The paint (built, checked: group A, 2026-09-25). Shaders `wgsl/cancel_paint.mjs` (`paintSource({ prec, B, blocksMax,
refSlots })`, `irowsSource({ n, ... })`, `ipicSource({ n, ... })`), host `back/paint.mjs` (`Paint.build(device, bh, {
log })` after `BackHalf.build`: the pipelines, PERMW, the uniforms, and `bh.cancel.stages` paint, irows, ipic
set; `lane(ln)` after `bh.lane`; `dispatch(p, ln, name)`), `bitmap.mjs permTable`. Every workgroup takes its slot's
size from PLAN's KEYTAB entry (SLOTS[r], the tag's one size bit; PLAN read-only, so the returns are uniform): the paint
serves every size from one pipeline, irows and ipic are built a size and return on a slot of another.
- PAINT, (blocksMax, 1, R), 128 threads: returns unless slot r is used, block b is set in the entry's bits and PTR
  names a record. The record's payload words into `packw` as the LDPC packed them (words 2 to 119, byte 472 alone in
  word 120), `crcOf` (CRC_SPLIT) gives the CRC, lane 0 writes it MSB first behind byte 472: the k = 3816 data bits
  (bytes MSB first, ldpc_encode's `data`). The codeword in workgroup atomics `cw` (159 words, a bit an index): data
  words first, then the 1272 parity bits: lane i takes checks 10 i .. 10 i + 9 in check order (chk = 12 i + r), each
  check's parity the XOR of its 15 data bits through the LDPC's `lay` (col z + (i - shift) mod z), the prefix over
  the ten kept a bit each; a Hillis-Steele XOR scan over the 128 lane totals gives each lane's exclusive prefix
  (ldpc_encode's running acc), parity bit K + chk atomicOr'd in. Then coefficient c of the block (2560, 20 a lane):
  slots 2c and 2c + 1 through PERMW (one u32 holds the pair), the whitening in bit 15, a slot past the codeword
  (0x7fff) carrying the whitening alone (map_out); `S[r sStride + UV[uvOff + 2560 b + c]] = (+-a, +-a)`, a =
  0.70710678, minus where the bit is 1 (focus_encode 616 to 620). Uniform: the LDPC's shape (sizes[s] = (blocks, UV
  entry offset, PERMW entry offset, 0), dims = (blocksMax, sStride, R, 0), lay, pw). Workgroup memory 1.7 KB.
- IROWS, (V, 1, R) a size, n / 8 threads: register k of thread tid holds x = tid + k T (unrolled): the disc row's
  entry at u = x (x < n / 2) or u = x - n where the row (pass 2's `rows` uniform: off, count, nonnegative entries,
  umaxN) holds it, conjugated, else zero; the forward kernels (`firstStage`, `laterStages`, the chain twiddles); the
  row stored conjugated in Y's slot at `r yStride + v n + x` (the C's fft_cols over the block array, focus.c 639).
- IPIC, (n / 4, 1, R) a size, n / 8 threads, pass 1's two exchange buffers: columns (x0, x0 + 1) and (x0 + 2, x0 + 3)
  as two packed inverses, register k at m = tid + k T from turn_pack's rows: m = 0 gives 2 Re A(0) + 2i Re B(0), m < V
  gives A(m) + i B(m) = (ar - bi, ai + br), m > n - V gives conj A + i conj B = (ar + bi, br - ai), zero between
  (A = Y[m][xa], B = Y[m][xb]); conjugated in, the kernels, conjugated out: column a the real part, column b the
  imaginary; clip_row's 0.5 + clamp(v, -lim, lim) 0.5 / lim with lim = 2 clip sqrt(0.5 subch 320) at clip 2, tilt 0
  (the uniform's f32, focus.c 646); the quad stored as one vec2u of pack2x16float pairs at `r picStride / 8 + (y n +
  x0) / 4`. The inverses are unscaled, as the encoder's.
- Checked by `scripts/gpu/back/test_cancel_paint.mjs` through the test page's "paint" job (the stages the back half wired, on the
  "back" build with cancel): for 12-00 (1024/256 bit map 0), 21-34 (1024/512 bit map 3), 02-36 (1024/256 bit
  map 0) and 06-26 (256/16 bit map 3), four frames' truth payloads (openRun `sent`) as REC records in a shuffled
  order with PLAN built on the host (`hostPlan`), slot 0 a random half of a frame's blocks, 1 all, 2 a quarter, 3 one
  block; a mixed batch of 12-00 and 06-26 (two PERMW modes); and an all-blocks frame at 1024/256 span 256. The disc
  (S's slot) equals `Focus.sent` bit for bit on every entry of every slot (+-a at a known block's coefficients, zero
  elsewhere: 4.7 M entries over the runs) on SwiftShader, the 4090 (f32) and the iGPU (f16). PIC against a JS f64
  picture of the same blocks (blockSlots, the coefficients at (u, v), the unscaled inverse along x then y over
  turn_pack's rows, clip): max abs 2.44e-4 (f16's half ulp at 0.5) and rms 1.1e-4 on every run in f32; on the iGPU
  f16 (the exchange buffers in f16 too) max abs 1.10e-3, 1.12e-3, 1.10e-3, 1.03e-3 (12-00, 21-34, 02-36, 06-26), rms
  2.4e-4, 0 samples over 4e-3, so the f32 exchange fallback the plan names is not built. The f64 reference against
  the wasm's own drive at 1024/256 span 256 (a pixel a sample from pixel 60 of 1144): max abs 2.29e-7, rms 4.4e-8;
  the device's all-blocks picture against that drive 2.44e-4 (f32), 1.16e-3 (iGPU f16).
- Coupling to flag: `paint.mjs` recomputed the soft stage's UV entry offsets and `Cancel.build`'s PERMW modes from
  the same `dims` and tables, since neither exported them; since 2026-09-27 it takes `bh.soft.uvOff`, and since
  2026-09-28 it makes PERMW and its offsets itself (the shift stage's for a day, section 14).

### 13.2 The gate (built): `wgsl/cancel_gate.mjs gate2Source`, one workgroup of 256 threads
Bindings: REC, BCOUNTS, LISTS (ro), PLAN, LISTS2, ARGS2, BCOUNTS2, CANCEL (rw), G (uniform, (n / CPW, V, blocks,
0) a size) and CARRY's header (uniform, 16 B at offset 0): 8 storage buffers. Phases, a `storageBarrier` and
`workgroupBarrier` between:
1. Everything empty: KEYTAB keys NONE, PTR 0, FR NONE, SLOTS NONE, BCOUNTS2 and CANCEL zero, LISTS2's counts zero
   and its (w, h) copied; each listed frame's size from LISTS.
2. Every record's key (id - block, from words 0 and 2) into KEYTAB by open addressing, hash (key / T) & 127 with T
   the size's blocks, `atomicCompareExchangeWeak` on the key word (a key of NONE is skipped: the one value no
   frame may have; a full table drops the record). Then, keys all in: the block's bit, PTR = record + 1
   (`atomicMax`), the count, the entry's size bit in `tag` (low 8 bits), KVER (`atomicMin` of the frame's block
   count), each frame's key range (workgroup min and max).
3. A thread a frame: short = listed and BCOUNTS[f][4] * 2 < best[size], best[size] the most any frame of that size verified
in the batch (a workgroup atomicMax after the sizes are known), or verified 0 where the batch's best is 0. It was
BCOUNTS[f][4] * 2 < T until late 2026-09-25, and verified < T at first; the relative form stops whole streams too dense
for the crop (every frame near half its blocks) from being cancelled for nothing (STATUS "Short is relative to the
batch"; section 13.4). fb = its own smallest key if it has records, else the
   greatest key of the nearest earlier frame of its size with records, else the carry's key (CARRY.valid, same
   size); fa = its own largest key, else the smallest key of the nearest later frame of its size with records.
   fb = fa = K: K + T if known (in the table under this size alone), else K - T, T the frame's block count (a
   painted frame's ids run over its version's blocks; the size's blocks until 2026-09-26). Each key marked USED in its
   entry's `tag` (a key touched by two sizes is never a reference; the carry key not in the table sets a flag).
4. Used keys ranked by key; ranks under R take paint slots (KEYTAB.slot, SLOTS[slot]); CARRY_SEL = the greatest
   slotted key's slot, key, count and size (NONE with none).
5. A thread a frame: fb and fa to slots (`slotOf`: the entry's slot if the key is in the table under this size,
   R if it is the carry's, else NONE, then dropped), nrefs, slotB = slotA when one; FR and CANCEL's header.
   Thread 0 then lists the frames with nrefs > 0 by size into LISTS2 and writes ARGS2 as the first gate does (pass
   2b's workgroups the most rows, LISTS' row counts, of the frames it lists, since 2026-09-26).

ZERO (`zeroSource`, (ceil(sVec4 / 256), 1, R)): S's frame slot r to zero when SLOTS[r] is not NONE. CARRY
(`carrySource`, (ceil(refVec4 / 4096), 1, 1)): REF slot CARRY_SEL.slot, bases and means, into CARRY behind the
header { 1, key, count, size }; nothing when no slot was painted.

### 13.3 The host (`back/cancel.mjs`, `back/back.mjs`, `decoder.mjs`; built, wired whole 2026-09-25)
- `Cancel.build(device, { B, tables, precision, code, dims, log })`: pipelines gate2, zero, carry; `gateUni`; the
  shared CARRY; `bytes` (a lane's, and CARRY); `plan` (`planLayout`); `stages` = { paint, irows,
  ipic, blur, bmeans, fit, solve }, a function (p, ln, picture) dispatching that stage; `wire(bh)` builds the
  paint (`Paint.build`, 13.1) and the bases and fit (`fit.mjs build` with the transform's `picUnis` when the back
  half samples the frame itself, 13.5) over the back half's transform and soft stage and fills every stage (one
  left unfilled fails the build); `BackHalf.build` calls it once every stage exists.
  `lane(ln)` makes PLAN, LISTS2, ARGS2, CANCEL, BCOUNTS2, PIC, REF, BPART, FITP (before soft.lane and ldpc.lane,
  which bind LISTS2 and BCOUNTS2 for their second groups); `bind(ln)` the gate2, zero and carry bind groups (after
  ldpc.lane: REC), then the paint's and the fit's (the fit over the lane's grid when it has one); `bindPicture(ln,
  picture)` and `bindGrid(ln, gridBuf)` forward to the fit; `prepare(enc, ln)` the carry in; `dispatch(p, ln,
  name, picture)` one of PASSES2 less SECOND ("paint" runs ZERO then the paint; the fit samples the batch's picture
  variant). Host builders for the stage tests: `hostPlan({ slots, frames })`, `hostLists(frames)`,
  `hostCancel(frames)`.
- The pass 1 cancel variant (`wgsl/back_transform.mjs pass1Source({ cancel: true, refSlot, refMeans })`, one a
  size, `Transform.build(..., { cancel: { refStride, refMeans } })`): since 13.7 it reads no picture but the
  samples pass 1 kept (KEEP, read-only at binding 0, `array<vec2u>`), then the grid variant's bindings 1 to 6, REF
  (read-only, binding 10, `array<vec2u>`) and CANCEL (uniform, binding 11, `array<{ hdr: vec4u, c0..c4: vec4f },
  B>`: the 24-word record): 7 storage buffers. Per workgroup the frame's CANCEL record: nothing is taken off a
  frame whose fit failed (ok 0) or that has no reference; per sample, before the column sums, `zz -= sum_j (a_j +
  c_j v + d_j u) (b_j(x, y) - bbar_j(y))` over slot A's three bases and, with two references, slot B's: one 24 B
  REF record a slot a row for the quad's four columns (three vec2u, `unpack2x16float`) and the row's means (f32
  behind the same view), u = x / (n - 1) - 0.5, v = y / (n - 1) - 0.5, the coefficients read from the record's five
  vec4f (slot A's nine first, then slot B's). The transform checks at build that `refMeans` is 6 nmax^2 (the
  three-base layout it reads): a REF laid out otherwise would fit nothing. It reads LISTS2 at binding 1 and counts
  into BCOUNTS2 at 5, dispatched indirect from ARGS2 (`dispatchPass1(p, ln, picture, true)`, the picture argument
  unused; the bind groups `p1cGroups` over the lane's KEEP, REF and CANCEL, made once by `bindSecond`).
- `BackHalf.build(..., { cancel })`, `bh.cancel`, `dims.cancel`; `lane()` orders the stages' lanes as above and
  the transform's second groups (`bindSecond`: reduce and pass 2 over LISTS2, counting into BCOUNTS2);
  `prepareCancel(enc, ln)` (the carry in, a copy, so on the encoder before the pass that holds pass two);
  `encodeCancel(target, ln, { ts, fused, frames })` (an encoder, a pass a stage, or an open pass behind pass one)
  runs PASSES2 over a lane whose pass one has run: gate2, paint, irows, ipic, blur, bmeans, fit, solve, pass1c,
  reduce2, pass2b, soft2, ldpc2, carry, the picture variant ("grid" or "fused") handed to every stage
  (`dispatchPass1(p, ln, picture, cancel)`, `dispatchReduce(p, ln, second)`, `dispatchPass2(p, ln, second)`,
  `soft.dispatch(p, ln, frames, "second")`, `ldpc.dispatch(p, ln, frames, "second")`); `readback`, encoded after
  it, copies BCOUNTS2 and CANCEL behind pass one's buffers (4 KB more at B 32), and `parseReadback` with
  `dims.cancel` gives `first` (pass one's record count: records from it on are pass two's) and a frame's `cancel` =
  { `counts2`, `refs` (nrefs), `cancelled` (nrefs > 0 and ok), `gained` (BCOUNTS2 column 4), `slots`, `ok`,
  `coef` }.
- `FrontHalf.create(..., { cancel })` -> `backOpts.cancel` (off by default since late 2026-09-25);
  `batch()` puts the carry in at the encoder's head and pass two behind pass one, in the batch's one compute pass
  (profiled: its 14 passes, `TS_MAX` 80), and gives each frame's `back.cancelled`, `refs`, `gained` with its records
  of both passes. The carry is made by `Cancel.build` inside `buildBack`, so `backShared` counts it and
  `destroyBack` frees it: a new back half (a plan, a lost device) starts with no carry. The Batcher's one-frame
  price builds the whole stage at B 1 (every pipeline compiled again), as it builds the rest of the back half.
- Records merge: REC holds pass one's records first (claimed by the LDPC's one atomicAdd on REC.count before pass
  two's dispatches start), then pass two's; the sum of BCOUNTS' record column is pass one's count, of BCOUNTS2's
  pass two's, and REC.count their sum. V and ITS hold the last verdict a block: pass two's LDPC returns at once on
  a block whose V is 1, so a verified block keeps pass one's verdict and iterations and is never recorded twice; a
  block pass two verifies goes from its pass one verdict (2 declined, 3 stall or cap, 4 CRC) to 1 with pass two's
  iterations. BCOUNTS is pass one's counters and BCOUNTS2 pass two's (columns 0 to 2 and 6 show pass 1c, pass 2b
  and soft2 ran on the frame; 3 the solve's k plus ldpc2's blocks tried; 4 = 5 the blocks gained; 7 ldpc2's
  iterations). A frame's blocks are its records of both passes; the harness scores every record against the truth
  stream, so a wrong gain would count as bad.
- The test page (`scripts/gpu/back/test_page.mjs`): the "back" kind takes `build.cancel` (pass two behind the job's last pass one, in
  its encoder; the result's `dims` carry `cancel` and `refSlots`, its `times2` pass two's stages); "paint" `{ build, rec: { off, len }, slots: [{ key, size,
  blocks: [[b, rec], ...] }] }` runs zero, paint, irows, ipic and returns S (R frame slots), PIC, PLAN; "fit" `{
  build, pics: [{ slot, off, len, key, size }], frames: [{ n, w, h, size, off, slotA, slotB, nrefs }] }` runs blur,
  bmeans over the slots, fit (over the lane's grid), solve and returns FITP, CANCEL, REF (R slots); each result's
  `layout` gives the byte ranges and `dims` the strides. Both kinds run the stages the back half wired.
  `scripts/gpu/back/fuse_page.mjs` takes `cancel` on a job: pass two runs in both of its back half runs (the carry in before each,
  `encodeCancel` behind `encode` on the one lane), so pass two behind the grid pass 1 and behind the fused one is
  compared on one batch's state.

### 13.4 Checked whole (`scripts/gpu/back/test_cancel.mjs`, 2026-09-25)
- The fit and solve dispatch direct, (n / 8, 1, B) and (B) with the kernel returning past LISTS2's count, as soft
  does: ARGS2 holds pass one's four slots only.
- What the test asserts, on grids from the C's registration (B 16, 09-14-21 frames 40 to 55 then 56 to 71, 02-36-59
  frames 0 to 15 then 16 to 31, each run on its own build so its carry starts empty): (a) the build without the
  stage and the build with it give the same pass one records and counters; (b) with it the records are a superset,
  every gained payload is the truth stream's, bad 0, gained above 0, every short frame with a decodable neighbour in
  the batch is cancelled, no frame gets the carry slot on a fresh build; (c) the next batch's leading short frames
  (before its first frame with records) get the carry slot R and the later ones do not; (d) test_back's controls
  on the cancel build gain nothing; (e) on the whole decoder (fuse_page, 09-14-21 frames 40 to 55 in find mode) pass
  two behind the grid pass 1 and behind the fused one (since 13.7 the one cancel variant over the samples each
  keeps; two variants before) gives bit-equal S over every frame slot and the same gains, bad 0.
- As run (B 16; 09-14-21 frames 40 to 55 hold four whole frames, the C's 128 blocks, 02-36-59's first 16 one):
  identical on SwiftShader f32, the 4090 f32 and the iGPU f16 for the grid batches: 09-14-21 cancel off 128 (the
  C's), on 210 (+82: frames 42, 48, 54 gain 26, 28, 28; the other nine short frames with references gain 0), 12 of
  12 short frames cancelled; its next batch (56 to 71) 151 with 55 gained (29, 1, 25 on frames 62, 66, 68), the one
  leading short frame (56) given the carry slot beside frame 57's; 02-36-59 32 to 64 (+32: frame 10 whole from
  0), 15 of 15 cancelled, its next batch 64 with 32 gained (frame 23), six leading short frames (16 to 21) given
  the carry; bad 0 everywhere, pass one's records and counters equal to the plain build's, the controls 0 (the
  noise and zeroed grids are given two references and gain nothing). The whole decoder on 09-14-21 frames 40 to
  55 (14 of 16 listed; the decoder's own registration, not the C's): 224 blocks with 96 gained, 10 frames
  cancelled, S bit-equal between the grid and fused cancel variants on SwiftShader and the iGPU f16 (2,621,440 of
  2,621,440 entries); on the 4090 498,016 entries equal, the rest within 2.83e-8, and two frames one block apart
  (222, 94 gained): pass one's own grid and fused samplers are not bit-equal on the 4090 (`test_fuse` there: 0.41%
  of S entries equal, -101 dB, blocks equal), so the test holds the cancel variants to bit-equality only where pass
  one alone is, and to 1e-6 elsewhere.
- The straddle-rich runs whole (the 4090 f32, B auto, `scripts/exp/gpu_captures.mjs FINDER=cnn FRAMES=600`, the dumps
  `results/gpu/perframe_*_cancel.tsv` against `_before_cancel.tsv`): 09-14-21 6,156 to 8,668 (+2,512, +40.8%: 484
  short frames, 444 given references, 444 cancelled, 142 of them gaining; two references on 369), 02-36-59 2,830 to
  4,271 (+1,441, +50.9%: 555 short, 521 cancelled, 90 gaining; two references on 323); bad 0, no frame lower, every
  gain on a frame with cancelled set. Above the plan's +25% target, so its NREFS_MAX = 3 knob was not tried. The
  iGPU's ms: pass two about 8.3 ms a frame at 1080, the fit most of it.

### 13.5 The bases, means, fit and solve (built, checked: group B, 2026-09-25)
Shaders `wgsl/cancel_blur.mjs` (`blurSource({ n })`, `meansSource({ n })`) and `wgsl/cancel_fit.mjs` (`fitSource({ n,
B, picture, grid })`, `solveSource({ B })`), host `back/fit.mjs` (`build(device, { B, tables, dims, grid, picUnis, log })`
giving `lane(ln)`, `bindGrid(ln, gridBuf)`, `bindPicture(ln, { texView, mapsBuf, residBuf })`, `dispatch(p, ln, name,
picture)` for `STAGES` = blur, bmeans, fit, solve; shaped as `soft.mjs`, for `cancel.mjs` to plug into its `stages`).
One uniform a size a lane serves all four (`PARAMS`: size, n, ns, st, then every stride in its buffer's own units:
picSlot f16 elements, refSlot and refMeans vec2u, bpartSlot and fitpStride f32, PLAN's FR and SLOTS word offsets,
gridStride, nwg = n / 8). Every buffer holding f16 is bound as u32 or vec2u and packed or unpacked in the shader, so
no stage needs `shader-f16`.
- Which slots: a paint slot r is processed by size s's pipelines when `SLOTS[r]` names a KEYTAB entry whose tag's
  size bits are `1 << s` alone (PLAN read-only, so the test is uniform and precedes the barriers). The carry slot is
  never blurred: it arrives complete.
- BLUR, (n / 16, n / 16, R) a size, 256 threads: the 26 x 26 halo tile of PIC (f16, the index clamped to the
  picture, which is the C's edge repeat) into workgroup memory; the horizontal pass makes 26 rows x 16 columns of
  the three sums (k7 on p, k11 on p, k7 on (p - 0.5)^2), the vertical pass one output a thread; 64 threads pack the
  quads into REF's records (`REF[refSlot r + ((x / 4) n + y) 3 + j]` = vec2u of pack2x16float pairs, base j), 16
  threads sum their row's 16 outputs a base into BPART (`bpartSlot r + j n (n / 16) + y (n / 16) + tx`, f32). The
  kernels are literals in the source (sigma 0.8: 7 taps, 1.6: 11 taps, normalised, `gaussian()` exported).
  Workgroup memory 10.8 KB.
- MEANS, (n / 256, 1, R): a thread a row folds the row's n / 16 partials a base and writes (bbar0, bbar1, bbar2, 0)
  as f32 bits through the same vec2u view at `refSlot r + refMeans + 2 y`.
- FIT, (`fitWorkgroups(n)` = n / 32, 1, B) a size and picture variant, returning past `LISTS2[s]`'s count (the
  shape since 2026-09-25, the cost round below; the build's shape is in the Cost bullet): workgroup (w, 0, z) takes
  the sampled rows j = 8w..8w + 7 of the z-th listed frame, y = 4 j (`ROW_STEP` 4, `ROWS` 8: every fourth row of the
  picture). Thread t of 256 samples x = st t + (j mod st), st = n / 256 (256 samples a row at every size, so a 256
  picture keeps every thread busy; the stagger follows the sampled row's index, since y itself is always a multiple
  of 4). The grid variant reads the strip-order grid (f32 or f16); the fused one binds the texture at 0, maps 7,
  resid 8, PIC 9 as fused pass 1 does and takes the sample through the same `sampleAt` (the column's and the row's
  curves taken a sample, since a thread's column moves with the row). A row: the 256 z into `red`, an 8-lane fold,
  zbar = sum / 256 (the row's samples, not the full row); then every thread writes its 19-vector, the 18 features
  (b_j - bbar_j) (1, v, u), slot A's at 0..8, slot B's at 9..17 (zero when nrefs is 1), and z - zbar at 18,
  sample-major in `F[t 19 + a]` (19 KB); then the products in registers: thread t holds the sums of `GROUP` = 95
  products (generated code, constant indices) over the two samples t mod 128 and t mod 128 + 128, threads under 128
  products 0..94 and the rest 95..188 (product p < 171 is pair p of the upper triangle row-major, p >= 171 the
  right-hand side entry p - 171, feature 18 the z - zbar; the branch is uniform over any wave of up to 128 lanes),
  so a row costs a thread 38 workgroup reads and 190 multiply-adds, where the shape before (thread t < 189 summing
  its one product over the 256 samples out of `F`) read two workgroup words a multiply-add. Four barriers a row.
  After the 8 rows the accumulators are folded through `F` in chunks of 16 products (each thread writes 16, 16
  threads a product sum 16 each into `red`, 32 threads write the two groups' sums; three barriers a chunk, six
  chunks) into `FITP[f fitpStride + w 208 + p]`.
- SOLVE, one pipeline, (B, 1, 1) a size over LISTS2, returning on nrefs 0: thread t < 189 sums its FITP entry over
  the frame's n / 32 workgroups into `A: array<f32, 18 x 19>` in workgroup memory (the triangle mirrored, rhs in
  column 18); k = 9 min(nrefs, 2). Gauss-Jordan with partial pivoting over columns 0..k - 1: thread 0 picks the pivot
  row (the largest |A[r][c]|, r >= c) into a workgroup word and sets `failed` when it is under 1e-6 (1 + |A00|),
  A00 read before any elimination; barrier; `workgroupUniformLoad` of the pivot row; 19 threads swap the rows;
  barrier; every entry off the pivot row reads its factor A[r][c] / A[c][c] and the pivot row's value; barrier;
  writes; barrier (the LDPC's flag pattern: a branch on plain workgroup memory would fail the uniformity analysis).
  After the loop coef[c] = A[c][18] / A[c][c] for c < k, 0 past k or on failure; `CANCEL[f].ok` = 1 or 0 and
  `BCOUNTS2[f][3]` = k (a store: ldpc2 adds after it). No private array anywhere.
- Checked by `scripts/gpu/back/test_cancel_fit.mjs` through the test page's "fit" job (the stages the back half wired, the grid
  fit over the lane's grid): two painted 1024 pictures (Focus.encode at 1024/256, span 256, where the drive holds the picture pixel
  for pixel at offset 60; 256 and 512 crops of them) in slots 0..4, captures z = 0.5 + s1 (b0_1 - 0.5) + s2 (b0_2 -
  0.5) + 0.05 sin(2 pi y / n) + N(0, 0.02), s1 = 0.3 + 0.5 v + 0.1 u, s2 = 0.7 - 0.5 v - 0.1 u, as f32 grids, B 8.
  SwiftShader f32 on the kept fit (every fourth row, the register shape; 2026-09-25): every base within 2.44e-4 of
  a JS f64 blur (f16's half ulp below 1; base2 within 6.1e-5), every mean within 1.9e-7, the means' pad 0; A and
  the right-hand side against a JS f64 accumulation on the same samples (the GPU's f16 bases and f32 means, the
  uploaded grid) within 3.6e-8 and 2.8e-8 relative (to the largest entry; 2.1e-7 and 1.8e-7 with the build's
  one-product-a-thread sums over 1,024 rows); the coefficients against a JS f64 Gauss-Jordan within 3.6e-6
  relative on the GPU's own A and 3.9e-6 on the JS A; the 1024 two-reference mix has its base0 planes within
  0.0073 and its other 12 coefficients within 0.0123 of 0 (0.0037 and 0.0077 on every row: a quarter of the
  samples, twice the noise), the single reference (k 9) within 0.0031 and 0.0051 with coefficients 9..17 exactly 0,
  the blur(p1, 1.6) mix recovered on base1 within 0.0091 (base0 of that reference within 0.005 of 0), the 256 mix
  within 0.0090 and 0.0161, the 512 single within 0.0042 and 0.0070; the degenerate frame (slotA = slotB, nrefs 2:
  its second nine columns equal the first) gives ok 0 and zero coefficients, as the f64 solve of the same A does;
  an unlisted frame keeps its header (NONE, NONE, 0, 0) and zero coefficients. The tolerances (0.01 planes, 0.02
  zeros) scale by sqrt(1024 / n) below 1024, since a size has 64 n samples. As built (every row), the 4090 (f32)
  gave the same numbers to the digit (A within 2.1e-7, solve within 1.4e-5, planes 0.0037, zeros 0.0077) and the
  iGPU (f16 build) the same fit (A within 9.3e-8, solve within 1.1e-5, planes 0.0037, zeros 0.0076) with its bases
  within 4.88e-4 of the f64 blur, one f16 ulp below 1 rather than half: its pack2x16float does not round to
  nearest; not re-run on the kept fit (the gates phase runs them).
- Cost, from the shapes: a reference's blur is n^2 / 256 workgroups of 676 loads and 3 x 25 taps a sample; a frame's
  fit is 64 n samples and 189 x 64 n multiply-adds (256 n and 189 x 256 n as built). On the iGPU (f16, 1080,
  `PROFILE=1`, 13.4's runs) a batch's blur costs up to 2.06 ms a frame and bmeans 0.09, the fit as built 3.46 to
  3.83 ms a frame and the solve 0.01: the fit was the stage's largest cost.
- The fit's cost round (2026-09-25; one iGPU timing a candidate, f16, 1080, `PROFILE=1`
  under the exclusive lock, `CANCEL=1 RUNS=09-14-21 FRAMES=96`, every other stage at the round's starting tree;
  the rule: a candidate losing more than 4 of the 440 gained blocks goes unless its ms saved is larger in
  proportion, one saving under 0.2 ms a frame is noise). The build: fit 3.84 ms a frame, compute 15.74, 1,016
  blocks. (1) Every fourth row (y = 4 j, the x stagger by j), n / 32 workgroups: fit 0.95, compute 12.81, 1,012
  blocks (4 of the gain lost, 18.6% of the frame's compute saved): kept. (2) On (1), the products accumulated in
  registers (the FIT shape above) instead of 189 threads reading two workgroup words a multiply-add: fit 0.71,
  compute 12.56, 1,013 blocks: kept; the fit is now about the 65,536 samples (the fused sampler, the REF reads) and
  the 32 barriers a workgroup, not the products. (3) On (2), the third base (blur((p - 0.5)^2, 0.8), the gamma term)
  dropped: k 12, a 16 B REF record, the blur one plane fewer: fit 0.63, blur 0.56, pass1c 0.79, compute 12.16, but
  997 blocks: 16 of the 440 gained lost (3.6%) for 0.40 ms (3.2% of compute), so deleted; the third base earns its
  keep (`test_cancel` with it: 09-14-21 +81 and its next batch 54, the decoder 222 with 94 gained, against +81, 56,
  223 with 95 without). Not tried: subgroup adds for the fold (the workgroup fold is a tenth of a workgroup's row
  work, so one path serves every device), fewer than 256 samples a row, other ROWS.
  `test_cancel` on the kept fit (SwiftShader f32): 09-14-21 frames 40 to 55 +81 (the build's +82), the next batch
  56 gained (55), 02-36-59 +32 and +32 unchanged, the decoder on 09-14-21's 16 frames 223 with 95 gained (224, 96),
  grid and fused cancel variants S bit-equal, bad 0, controls 0.

### 13.6 Pass two's stop rule (2026-09-25)
The second LDPC dispatch (`ldpc2`) runs on the short frames' cancelled spectra, where most codewords still fail:
under pass one's funnel it cost 2.14 ms a frame on 09-14-21's first 96 (iGPU f16, 1080, `PROFILE=1`), the fit
apart the largest cost of the stage. `ldpc.mjs build` now compiles two pipelines from the one source, pass one's
under `stop` (the funnel then, the learned rule since 2026-09-28, or a page's `?ldpcstop=`) and pass two's under `ref_ldpc.mjs DEFAULT_STOP2` = `"funnel2"`,
which nothing overrides; `dispatch(..., second)` picks the second. Pass one is untouched: `test_ldpc` gives the same
table (0 of the C's blocks given up, verdicts and iterations equal to the reference on every block).

The rule was set as the first funnel was (section 6), on the codewords pass two tries. 576 frames of the two
straddle-rich runs (09-14-21 and 02-36-59, frames 0 to 287 each) went through the cancel build on the C's grids in
batches of 16 (a scratch page reading L and BLK back after the batch), and every block on LISTS2 that pass one had not
verified and soft2 had not declined went through `ref_ldpc.mjs makeTracer` to the cap with no rule: 15,494 codewords,
1,822 of them gained on the GPU (the tracer's iterations to clear equal the GPU's ITS on every one). A rule that reads
only the count of violated checks is scored exactly from the traces.
- A lower cap loses blocks: gained blocks clear as late as iteration 30 (cap 20 gives up 56 of 1,822, cap 25 loses
  on both runs). The cap stays 30.
- The clearing codewords keep a lower running minimum than pass one's population: the highest running minimum among
  blocks that go on to clear past the checkpoint is 384, 369, 331, 331, 331, 264, 243, 119 checks after iterations 1,
  2, 4, 6, 9, 12, 15, 20 (the funnel's checkpoints are 446, 408, 401, 369, 344, 331, 287). The failures' median at the
  same points is 429, 405, 390, 382, 375, 370, 367, 362: near 0.3 m throughout, above the first three checkpoints, so
  a failing codeword ran to 9 or beyond under pass one's rule.
- `funnel2` = (1, 0.322), (2, 0.311), (4, 0.281), (12, 0.228), (15, 0.212), (20, 0.114): counts 410, 396, 358, 291,
  270, 146, each 0.02 m (26 or 27 checks) above that highest running minimum, the margin section 6 took as the stand-in
  for a held-out set. The checkpoints at 6 and 9 go: a running minimum is non-increasing, so a count equal to the
  one at 4 fires there or never. On the traces it gives up none of the 1,822 and runs 54.6% fewer iterations (53.1%
  on 09-14-21's first 96, 61.4% on 02-36-59).
- Timed on the iGPU (f16, 1080, `PROFILE=1`, `CANCEL=1 RUNS=09-14-21 FRAMES=96`, the exclusive lock, no other
  harness Chrome alive, on the tree the 15.74 baseline was taken on with this change alone): ldpc2 2.14 to 1.13 ms a
  frame, compute 15.74 to 14.77, blocks 1,016 of 1,016 (every other stage at its baseline: fit 3.88, pass1c 0.86,
  blur 0.80, ipic 0.27, ldpc 2.10). A margin of 0.01 m (13 checks: 397, 382, 344, 278, 256, 133) gave ldpc2 0.96,
  compute 14.60, the same 1,016 blocks: 0.17 ms a frame more, under the 0.2 ms taken as noise, for half the margin,
  so it was not kept. Not timed: the 0.015 m margin between them.
- Checked: `test_ldpc` on the final tree gives the table before this change (0 of the C's blocks given up, every
  block's verdict and iterations equal to the reference); `test_cancel` on SwiftShader on the tree the timings ran
  on gives 13.4's rows unchanged (09-14-21 128 to 210, +82, the next batch +55 with the carry on one; 02-36-59 32 to
  64, +32, the next +32 with the carry on six; the whole decoder 224 with 96 gained, grid and fused S bit-equal;
  bad 0, controls 0). The traces and the scratch page that read L back were not kept in the tree: a new recording
  that wants the rule set again needs them made a proper `exp/` script (as `ldpc_stop.mjs`).

### 13.7 Pass 1c reads the samples pass 1 kept (2026-09-25)
Pass 1c sampled the frame again through the map (toImage, coonsMix, bilin0 a sample: pass 1's own arithmetic),
since the fused pass 1 stored no grid: 0.86 ms a frame on 09-14-21's first 96 (iGPU f16, 1080), and the sampling
is 0.56 of a fused pass (section 12). Now a transform built with `cancel` has either pass 1 variant keep every
sample it read in KEEP (`Transform.lane` makes it, counted in `ln.bytes` so the Batcher prices it): unorm16
(`pack2x16unorm`; a sample is luma in [0, 1], the r8unorm texture and the C's lut, held to 7.6e-6; f16 holds it to
2.4e-4 near 1 and moved a block on `test_cancel`'s batches; f32 would be 4 MB a frame), quad-major, vec2u (four
adjacent x) at `f sampStride + (x / 4) n + y` with `sampStride` = nmax^2 / 4 vec2u a frame (Params word 10), so a
workgroup's rows are one contiguous 8 B store and read a thread. The cancel variant is one pipeline a size over
KEEP (binding 0, read-only), whatever picture the batch used; the fused cancel variant and the grid one, each
re-reading its picture, are deleted with `bindCancelGrid`/`bindCancelPicture`, and `bindSecond` makes `p1cGroups`
once. Storage buffers a stage: the grid pass 1 7, the fused 8, pass 1c 7.
- Timed on the iGPU (f16, 1080, `PROFILE=1`, `CANCEL=1 RUNS=09-14-21 FRAMES=96`, the exclusive lock, no harness
  Chrome alive), a pair on one tree (the fit of 13.5 at 0.70 ms and 13.6's funnel2 in it, every other file
  checksummed unchanged through both runs): pass1c 0.84 to 0.39 ms a frame, pass 1 0.75 in both (the write costs
  nothing the timestamps see), compute 11.27 to 10.86, blocks 1,013 to 1,010 (437 to 434 gained on 73 cancelled
  frames, under 1% of the gain), a lane 317 to 381 MB (lanes of 32, 2 MB a frame). The three blocks are the
  rounding to 1/65535: a knife-edge block moves either way (on `test_cancel`'s batches f16, unorm16 and an f32 KEEP
  gave +81 / +56 / +32 / +32 on the same tree, the decoder check 223 to 225). The round's earlier timing of this
  change, 10.26 ms with 0 gained, ran while the fit's files were swapped under it (a two-base REF against
  a three-base read) and counts for nothing; the transform now checks REF's layout at build (`refMeans` = 6 nmax^2,
  the three-base record it reads), so a swap like it fails the build instead of fitting nothing.
- Checked on the final tree: `test_cancel` on SwiftShader and on the iGPU f16: pass one unchanged, +81 on 09-14-21's
  40 to 55 and +32 on 02-36-59's first 16, the carry (56 on one leading short frame, 32 on six), controls 0, grid
  and fused S bit-equal (2,621,440 of 2,621,440) with 224 (SwiftShader) and 225 (iGPU) blocks on the decoder check;
  the block of 13.4's +82 is the fit's new sample set (an f32 KEEP gave +81 on the same tree), not this change.

### 13.8 Pass two deferred a batch (late 2026-09-25, reverted 2026-09-27)
From late 2026-09-25 batch N's pass two ran in batch N + 1's command buffer on N's lane, so that a device which
overlaps independent compute could run it beside N + 1's pass one. None did: in Chrome 153 (Dawn on Vulkan)
independent dispatches, passes and submits ran one after the other on the 4090 and the iGPU (`archive/build-scratch-2026-09/overlap_bench.mjs`:
two took 2.0 times one; the decoder's `overlapMs` read 0; STATUS "Whether any overlap happens"). And nothing of N + 1
fed N's cancellation: its references are its own records and the carry (13.2), which pass two in N's own command
buffer reads the same way. Reverted on 2026-09-27: the deferral was useless, and a second pass in the batch's own
command buffer is simpler. Gone: the owed lane (`root.prev`, `fh.owed`), the flush (`run([], null, { cancelOnly:
true })`), the late readback (`readbackLate`, `lateBytes`, 1.04 MiB a batch of 32 at 1024/560; `parseLate`,
`out.late`), `overlapMs` and the second timestamp pair, the batches the receiver's queue and the harness page held for
the next readback, the worker's late result. At the revert both forms gave every frame the same blocks on
`gpu_front 16` (4090, cancellation off and on) and on 07-56-38 with it off; with it on, 07-56-38 moves a few frames
run to run in either form (a file: KEYTAB fills, 13.2), and the iGPU's cost matched (2026-09-27).

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
  the cancel stage, is the paint's (13.1).
