# The GPU back half: design

Input: each frame of a batch (B frames, any picture size each), sampled through the front half's map by the fused
pass 1 (section 10). Output: CRC-verified blocks with their ids, read back once a batch, with a verdict and an
iteration count a block, each block's pilot reading and each frame's grid shift. The reference for every rule is
`src/lizard_phy.c` `lizard_finish_ext`, `src/ldpc.c`, `src/layout.c` `ob_crc32c`. The decoder is judged on blocks that pass
their CRC-32C against the truth stream, on ms a frame at batch B (the iGPU's; the 4090 counts blocks), and on a
negative control; never on parity with the C. The tables named below come from the wasm codec (`init()` from
`sim/ob.mjs`, section 8).

Every block is 476 payload bytes, the first 4 the block id (LE). The pictures are 384, 512, 768, 1024, 1152, 1280 and
1536 (`plan/common.mjs` `SIZES`, the sizes of `sim/lizard_pick.mjs` `PICTURE_SIZES` a format has; tops 80, 128, 320,
560, 720, 880 and 1024 sub-channels). The code
rate follows the
frequency (`src/lizard_phy.h` `lizard_tiers_for`): from the lowest frequencies out, 7/8 blocks of 4496 slots (n 4608, 4480
sent, then 16 pilot slots), 3/4 of 5152 (n 5120, k 3840, m 1280, 10 block rows of 13 or 14 data slots, then 32 pilot
slots), 2/3 of 5760 (n 5760, 15 block rows) and 1/2 of 7680 (n 7680, 30 block rows), every code at z = 128, the blocks
packed one after another from the frame's first slot whatever the sub-channels, the slots past the last block padded
(the whitening alone). The soft values and the LDPC are the rule stage (`back/tiers.mjs`;
the kernels `glsl/back_tiers.mjs`, TAB's layout and the codes' shapes `plan/back_tiers.mjs`): a frame's profile is a row of its table by the frame's version, and the arithmetic of sections
1 and 4 to 7 runs a code at a time, each code with its own block rows and its own learned stop. Where a section gives
one code's numbers, they are the 3/4 code's.

## 1. Stages, in order, and the arithmetic each reproduces

The picture z(x, y) is luminance in 0..1 (the C's gamma table at gamma 1 is i / 255; the sampler reads r8unorm).
Everything after the picture is scale free except the int8 quantum of the soft values, so a constant factor on z is
invisible. The spectrum is carried as S = T / n^2 (T the C's unscaled transform, which reaches past f16's 65504);
every constant of the detrend is divided the same way at the store.

- **A0 reduce**, per frame. cx[i] = (2 i + 1 - n) / n, qx[i] = cx[i]^2 - mean(cx^2), sx2 = sum cx^2, sq2 = sum qx^2
  (all from `lizard_tables`). Five sums over the picture: Sx = sum z cx[x], Sy = sum z cx[y], Sxy = sum z cx[x] cx[y],
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
  pyy X2[v]). X1, X2 are the DFTs of cx and qx (`lizard_tables`, complex, index (u + n) mod n). T(0, 0) carries
  nothing and is not stored. This is the C's spectral subtraction (lizard_phy.c `lizard_finish_ext`) to the letter; a
  spatial subtraction of the same surface before A1 is identical by linearity, but it costs the picture's bytes again
  and breaks the fused sampler, which has no picture to pre-pass.
- **B soft values**, per sub-channel j of the frame over its 320 coefficients y_s = S at pos[320 j + s]: m2 = mean |y|^2, m4 = mean |y|^4,
  a2 = sqrt(max(2 m2^2 - m4, 0)), nv = max(m2 - a2, 0.02 m2), k = 0.7 * 2 sqrt(2 a2) / nv, LLR = clamp(k y, -10, 10)
  per axis, int8 = round half to even (WGSL `round`, the C's `nearbyintf`) of 8 LLR; out[2 s] the real axis,
  out[2 s + 1] the imaginary: slot order, the C's `ext_llr`. est_j = J(a2 / nv), J(g) = (1 - 2^(-0.3073 (2 sqrt
  g)^1.7870))^1.1064 for g > 0, else 0. blk_est = the block's sub-channels' est, each by the slots of the block it
  holds, over the block's slots (a block's slots run across sub-channels as they fall: the C's `lizard_finish_ext`). f32
  throughout (the C: double). 2 m2^2 - m4 is a difference of near equals only where nv is at its floor, a block that
  declines anyway.
- **B decline gate**: declined = blk_est < 0.2 k / nt (`LIZARD_DECLINE`; 0.15 at 3/4). A block within 1e-6 of the bar
  may differ from the C. Accepted.
- **C bit map and whitening** (`map_in`): codeword bit j of block b is slot inv[j] of the block (from the block's
  first slot), sign flipped where white[slot of the frame] = 1. perm[i] = i P mod nt (`perm_fill`, LINEAR, the format's
  one bit map: P = 1957 at the 3/4 code's 5120), inv = perm^-1; white = PRBS-23, x^23 + x^18 + 1 from the all-ones state, one bit a slot of the
  frame in pos order (`whiten_fill`). The slots past a block's codeword carry the pilots (the whitening XOR a bit of
  the painted picture's count, SPEC 7.3), which the soft stage reads and fits the grid's shift from (32 such slots at
  3/4, 16 at 7/8, none at 2/3 and 1/2, whose codewords fill their blocks). 476 * 8 + 32 = 3840 bits, the k of the 3/4, 2/3 and
  1/2 codes; the 7/8 code's k is 3968, and its bits past 3840 go in as known zeros at 127, as the C's do.
- **C LDPC**: normalised min-sum, layered by block row, norm 13/16, posteriors clamped to +-8191, messages to
  +-127, cap 30 iterations, sweep direction alternating each iteration, exit on a clear syndrome. Integer
  arithmetic, kept integer: the norm and caps were set on 78 000 blocks and mean what they were measured to mean only
  in integers. The C gives up on a failing codeword by its stall rule (violated checks after iteration 9 above 0.95
  of the count after iteration 1); the GPU by each code's learned stop (section 6), which gives up by a small net's
  reading and verifies the same codewords but for its budget of 1 in 1,000.
- **C CRC-32C and compaction**: the 476 payload bytes are systematic bits 0..3807, MSB first in each byte; the CRC
  is bits 3808..3839 MSB first; `ob_crc32c` is CRC-32C, the reflected polynomial 0x82F63B78, init and final xor
  0xFFFFFFFF.
  Equal means verified; a verified block claims a record slot and is written; nothing else leaves the device.

Drift is acceptable everywhere before the int8 quantisation and in the quantisation itself: a slot within f32
noise of a rounding boundary may differ by one, and a marginal block may then decode or not. The acceptance
criterion is blocks against the truth stream on the same frames, not equal bits. Not in this design: a
decision-directed second pass and burst accumulation. The GPU reads every frame once.
## 2. Buffers between stages

Frame f of a batch of B (frames may differ in picture size; every stage reads `sel[f]` and `frames[f].valid` as
the sampler does). `sel[f]` is a vec4u: (ring, 1 + version, picture slot, 0). PICK (F6) writes (ring, found, 0, 0);
F8 (`gpu/glsl/word.mjs`) then (ring, 1 + v, slot, 0) for the version v its word or the batch's held word names, else
y 0. A frame is sampled and listed when y >= 2; z indexes the picture sizes.

| name | writer -> reader | element and order |
|---|---|---|
| LISTS, ARGS | gate -> all | u32 [SLOTS sizes][1 + B] (SLOTS = 8, `plan/common.mjs`): count, then frame indices, then each frame's (w, h), each frame's version (the one F8 read in sel.y = 1 + v, or the held one; a frame with neither is not listed; the rule stage runs that version's profile), each frame's row count (`listsRows`: the disc rows that count takes, section 3 "Rows a frame"), then each frame's ring (`listsRing`), then every listed frame (`listsAll`); u32 [SLOTS][12] indirect args, (x, y, z) for pass 1, reduce, pass 2 (the most rows a listed frame of the size takes) and tsoft (the chunks of the size's most listed version; talign and tblocks on reduce's), then u32 [4][3] each code's LDPC (`argsLdpc`: the code's most blocks over the listed frames' versions, the listed frames, 1); (SLOTS (1 + B) + 6 B) words, `listsWords`; 432 B, `argsWords` |
| PART | A1 -> reduce, A2 | f32 [n][4] a column: (sum z, sum z cx[y], sum z qx[y], the column mean) |
| COEF | reduce -> A2 | f32 [8]: px, py, pxy, pxx, pyy, 0, 0, 0 |
| Y | A1 -> A2 | complex, v major: (v, x) at (v - 1) n + x, 1 <= v < V; f16x2 (f32x2 on the 4090); V rows of room a frame |
| S | A2 -> B | the disc, one entry a used coefficient, row compact (section 3); f16x2 or f32x2 |
| L | B -> C | int8 [subch][640] packed 4 a u32, slot order (the C's `ext_llr`) |
| EP, BLK | tsoft -> tblocks -> tldpc | EP (est, pilot) a sub-channel; BLK a block: (f32 est, u32 declined) |
| PILOT | soft -> host | f32 [blocksMax] a frame, each block's pilot reading; then two f32 a frame, its grid shift |
| V, ITS | C -> host | one 32-bit word a block each, at f * blocksMax + b: u32 verdict (0 not run, 1 verified, 2 declined, 3 stall or cap, 4 CRC), i32 iterations (the C's codes) |
| REC | C -> host | u32 count, then 484 B records: word 0 = frame + 65536 block, word 1 = iterations, 476 payload bytes (id first); cap = blocks in the batch |
| BCOUNTS | all -> host | u32 [B][8]: columns transformed, rows transformed (the frame's rows), coefficients stored, blocks tried, verified, records, sub-channels quantised, iterations run |

S holds subch * 320 entries exactly (verified against `npos`), under the C's half plane (vblocks * n * 64). V and
ITS are a word a block, since WGSL has no byte store (section 7), and a frame's rows sit at the largest built size's
blocksMax. The host copies the whole of REC in the batch's one readback, because the count is known only on the
device (section 9).
## 3. The transform

Coefficients needed: the half disc u^2 + v^2 <= R^2, v >= 0, without v = 0 with u <= 0, R = sqrt(640 subch / pi),
taken in order of r^2, then v, then u, the first subch * 320 (`init` in lizard_phy.c), enumerated from `lizard_pos_ptr`.
A frame needs the rows v < V = vmax + 1, vmax = floor(R) (umax(0) the same).

S layout: row v at off[v], v = 0 holds u = 1..umax(0), v > 0 holds u = 0..umax(v) then u = -umax(v)..-1, with
umax(v) = floor(sqrt(R^2 - v^2)) capped so the count is exactly subch * 320 (`discLayout` in `disc.mjs`
builds off, umax and the coefficient-to-entry table from pos; every entry of pos maps to exactly one entry of S).

Order: along y first, then x, as the C does. Both orders cost the same FFTs: n / 2 real pairs of length n, then V
rows of length n (x first keeps u = 0..umax of each real row and recovers u < 0 from T(-u, v) = conj T(u, -v)). With
the sampler fused into pass 1 the picture is not read at all, and y first moves less than x first, whose pass 2
would read Y at a stride. Output pruning inside a pass is worth at most the last stage (the band holds every residue
until 2^s exceeds 2 umax + 1) and is not done; the pruning is which rows exist (V of n / 2) and what is stored (the
disc).

Kernel shape: one column pair in workgroup memory, Stockham, eight points a thread in registers, one exchange
through workgroup memory a stage, no ping-pong buffer. Threads a pair = n / 8. Stages (`stagesOf`): 256 = 8 8 4,
512 = 8 8 8, 1024 = 8 8 8 2, 384 = 8 8 2 3, 768 = 8 8 4 3 and 1536 = 8 8 8 3, the radix-3 stage last and in place
(`radix3Stage`). Any 8 x 2^a 3^b 5^c takes its odd stages after the powers of two, the last in place (`radix3Stage`,
`oddLastStage`), an earlier one through registers across the barrier (`oddStage`): the sender's inverse at 1152 =
8 8 2 3 3 and 1280 = 8 8 4 5; the decoder reads both pictures at 1024 (section 10). Workgroup memory a pair: n * 8 B in f32, n * 4 B in f16. A workgroup takes 4 adjacent picture
columns (2 pairs). Pass 1 writes Y with x contiguous in runs of 4 (the L2 merges neighbouring workgroups' partial
lines). Pass 2 reads a row of Y contiguous (the v = 0 row from Y0 as f32) and writes its compact disc row. A pipeline
a pass for each of the eight n, dispatched indirectly from ARGS over LISTS; a workgroup for a frame of another size
returns before its first barrier.

Rows a frame. Every size is built at its top (`gpu/tables.mjs` PICTURES) and a frame decodes the version its word
names, a prefix of the top's coefficients, so it needs only the rows its blocks' coefficients reach. `disc.mjs
rowsByCount` gives rowsBy[c], the rows the first c 8-sub-channel chunks take (chunk b is pos[2560 b .. 2560 b +
2559], sub-channels 8b to 8b + 7, whatever slots the rate profile's blocks take; a version is two chunks, 16
sub-channels),
from the top's own pos table, and `rowsOfVersion` a version's, rowsBy[2v]; `scripts/gpu/back/check_rows.mjs` holds
every version of every size to its own disc's V. The gate's uniform carries it as rows[v][size], a row a version; the
gate writes each frame's rows beside its version in LISTS
(`listsRows`), pass 1 stores Y rows 1 to rows - 1 (the FFT itself is unchanged: every column still runs whole), pass 2
returns on a row past the frame's before its first barrier (LISTS is read-only there, so the branch is uniform) and
its indirect dispatch takes the most rows any listed frame of the size needs. S past a frame's rows is left stale:
the soft stage reads only its own blocks' coefficients, all inside them. Every frame decodes on its own, with no
memory across frames or batches, so the word is the only source: a frame whose word is not read decodes at the
batch's held word, or, with none held, is not listed. irows and ipic (the sender's inverse transforms,
`wgsl/inverse.mjs`, the sender's) run the size's V: a symbol's rows past its version are zero.

Precision: f16 for Y, S and the workgroup-memory exchange where `shader-f16` exists (the iGPU, phones), f32 on
the 4090; f32 for the butterfly arithmetic, the partials, the reduce and the detrend epilogue. Why the means
come out first: the picture's mean is about 0.5 against a data rms near 0.125 and a single coefficient near
3e-4 of n^2; carried through f16 stages, the mean's rounding (relative 2^-12 of the sub-DFT DC at every stage,
landing on bins 0 and 2^s) sits within about 25 dB of the data on the v = 0 row and about 37 dB at v = n / 4.
With each column's mean taken out ahead of pass 1 and the v = 0 row kept in f32 through both passes, the f16
floor is about 50 dB under the strongest ring, and blur takes the outer rings 20 to 30 dB down from there.

As built (`back/transform.mjs`, `back/disc.mjs`; the kernels `glsl/back_transform.mjs`, the layouts `plan/back_transform.mjs`):
- Per frame strides are the largest built size's: `t.sStride` (S entries), `t.yStride` (Y entries, V n), `t.partStride`
  (the column means ride in PART's w, so there is no Y0); `t.cbytes` is 4 (f16x2) or 8 (f32x2). Frame f's disc starts
  at `f * t.sStride` entries into the lane's S; row v of it at `tables[size].off[v]`, entry e of the row is u = e for
  e < np (u = e + 1 on v = 0), else e - np - umaxN, with `rows[v] = (off, count, np, umaxN)`. `tables[size].UV[i]` is
  the entry of pos[i], and `UC[i]` its (u, v). The rows table is bound as a uniform (V <= 512), so pass 2 holds to 8
  storage buffers.
- LISTS is u32 [SLOTS][1 + B], count then frame indices, size major, then the frames' fields (section 2). ARGS is u32
  [SLOTS][12] (pass 1, reduce, pass 2, tsoft) with usage INDIRECT, entry k of size s at byte offset 4 (12 s + 3 k)
  (`ARGS_SLOT`, `argsOffset`), then each code's LDPC dispatch (`argsLdpc`, `ldpcArgsOffset`). The gate lists a frame only when frames[f].valid, sel[f].y >= 2 and sel[f].z names a
  built size below `SLOTS`.
- COEF is f32 [B][8] = (px, py, pxy, pxx, pyy, 0, 0, 0). BCOUNTS u32 [B][8], zeroed by the gate (with V, ITS and REC's
  count); columns 0 to 2 written by the transform (columns transformed = n, rows transformed = the frame's rows,
  entries stored = npos), the rest by the rule stage.
- The stored S is T / n^2 as designed. In f16 the smallest disc coefficients of a blurred frame land in f16's
  subnormals; the rms error over the disc is set by the strong coefficients. A scale, should one be wanted, is one
  constant at pass 2's store, and the soft values are scale free.
- Pass 1 takes a workgroup's four columns as one vec4f load a row and transforms the quad's two pairs side by side
  (two exchange buffers, one set of twiddles and barriers), so Y(v) for the four columns leaves as one 16 B store
  (f16) or two `vec4f`. A scalar load a column a row is bound by the cache lines a wave's load touches; a wider tile,
  a twiddle table or sincos do not help that. Twiddles: the base twiddle from the table, the rest by multiplication.
  Workgroup memory, at the size pass 1 runs (next bullet): two exchange buffers plus 16 B a thread of partials and
  the anchors' 460 B (`pass1WorkgroupBytes`): for the 1536 picture, run at 1024, 10,700 B in f16 and 18,892 B in f32,
  so a 32 KB device builds every picture in either.
- Pass 1 reads a picture at the ladder size below its own (`disc.mjs` `TRANSFORM_DOWN`): n is the first ladder
  size of at least 3 R, so the one below holds 2 R and every coefficient the picture carries. 1152 and 1280 read at
  1024, as 1536 does: 1024 holds 2 R for both sizes' tops, and 1152 would take pass 1 a third wave of 64 lanes.
## 4. Soft values

Formula and moments: section 1. The rule stage's soft pass, a dispatch a picture slot each (`glsl/back_tiers.mjs`):
- **talign**, a workgroup a frame, a thread a block: the grid's shift by the pilots (each block's own tail against its
  known symbols and the block's sign from its own pilots' sum, zi = k zr (u dx + v dy) fitted by least squares twice,
  kept where it stands `ALIGN_CHI` over the fit's residual variance), to the end of PILOT. A thread gathers its tail
  once, straight-line (its index loads, then its S and UC loads); the first fit is about a zero shift, the second
  about the first's answer (the tail turned); each fit's seven sums over the 128 threads on 16 threads, then one.
- **tsoft**, a workgroup an 8-sub-channel chunk of a frame, 256 threads = 8 sub-channels x 32 lanes x 10
  coefficients: gather a sub-channel's 320 coefficients from S through the UV table (u32 an entry, the S index of
  pos[i]; subch * 320 entries), each turned back by the frame's shift through UC, reduce m2 and m4, compute a2, nv,
  k and est, quantise every coefficient to two int8 into L, and the sub-channel's part of a block's pilot reading
  where a tail meets it (`tailIn`: at most one tail a sub-channel, blocks being 4496 slots or more; a tail may run
  across two); est and the part to EP. Lane l takes the coefficient pairs 2 p, 2 p + 1 for p = l + 32 k (k 0 to 4), so a
  pair is one L word and the lanes' UV, UC and L accesses are consecutive; the kernel is straight-line (every index
  load, then every S load, then the turn under one branch on the frame's shift). Each value takes its slot's
  whitening (a pair's four slots are four bits of one WHITE word), so L holds the whitened values; a pilot part is
  the sum of the tail's whitened values in the sub-channel over its rms axis value sqrt(m2 / 2), the sum reduced with
  m2 and m4 in one tree (0 where no tail meets it or m2 is 0). Dispatched (chunks of the
  slot's top, 1, B); a workgroup past the frame's chunks (two a version) returns before its first barrier.
- **tblocks**, a workgroup a frame: a block's estimate (its sub-channels' est, each by the slots of the block it
  holds, over the block's slots), its decline under its code's bar (section 5), its pilot (the parts of the one or two
  sub-channels its tail meets, over the tail's slots; NaN for a 2/3 or 1/2 block), to BLK and PILOT.

L stays in the C's slot order, whitened: L[slot] = (white[slot] ? -1 : 1) * llr[slot], WHITE a bit a slot of the
frame over 1024 sub-channels in TAB. The LDPC reads the bit map at its gather, from TAB: a code's INV (codeword bit
j's slot of the block, two u16 a word, 0xffff for a bit never sent: the 7/8 code's 128 known zeros, which the gather
sets to 127 without reading INV), its loads
L[block's first slot + inv[j]] (the block's first slot a multiple of 16, so L's word (slot0 + x slots) / 4 on).

## 5. The decline gate

BLK.declined a block from blk_est under its code's bar, 0.2 k / nt (`LIZARD_DECLINE`; 0.15 at 3/4). The LDPC's
workgroup for a declined block writes verdict 2, ITS -2, and returns before touching L: a workgroup with no work,
which the shape rules allow. The gate is a format decoder rule the C sets, measured on 78 000 blocks; on hard
captures the iterations it saves are among the LDPC's largest costs.
## 6. LDPC

The rate profile's kernel (`tldpcSource`, `glsl/back_tiers.mjs`; a dispatch a code: (the code's most blocks in a
served version, frames)) runs layered min-sum a code at a time: the code's n, k, z and slots are the module's
constants, its block rows (unequal at 2/3 and 1/2) and data slots (col | shift << 16) come from its uniform, its bit
map from TAB over L whitened (section 4), the bits past the 476-byte payload and its CRC (3,840: at 7/8 alone) are known zeros at
127, and each code gives up by its own learned stop (below). The check records are each lane's own (lane i is check
i of every row), the posteriors two i16 a word (updated by atomic XOR, since two lanes may write the two halves of
one word in a layer), an i32 a word where 32 KB still holds them, or an int16_t each with plain stores (where the
device has shaderInt16; the array declared in whole 32-bit words), and the violated checks are counted after every
iteration from a bitmap of the posteriors' signs. The three forms differ in that store alone and decide alike; the
native host times them (`LDPC_FORMS`) on its first batch and keeps the fastest, leaving out a form whose kernels the
device's workgroups cannot hold or whose module needs a feature the device lacks (a pipeline's `needs`).

H is never stored. The device holds each code's block row tables from `lizard_ldpc_tables`: lay_ptr, then the
(column, shift) pairs; at 3/4 each of the 10 block rows has 13 or 14 data slots and 2 staircase slots. Check (r, i)
reads data slot e as codeword bit col_e * z + (i - shift_e + z) mod z, its own parity bit k + mb i + r, and the
previous check's k + mb i + r - 1; for r = 0 the previous is (mb - 1, i - 1), and at i = 0 it is absent (the C models
it as a certain zero, posterior 8191, message re-zeroed every layer, and never scatters it back; skipping the slot is
the same arithmetic). A JS encoder built from exactly these tables reproduces what the C paints on every slot
(`scripts/gpu/back/check_tiers.mjs`, every code), so the H rebuilt from these tables is the C's.

Schedule, per iteration: rows r = 0..mb - 1 (odd iterations mb - 1..0). Per row, lane i: for each of the row's slots
q_e = L[idx_e] - R_e; track min1, min2, arg, sign parity; m1 = min(127, (min1 * 13) >> 4), m2 likewise;
R_e = (parity xor sign(q_e)) ? -m : m; L[idx_e] = clamp(q_e + R_e, -8191, 8191); keep the check's record. Within a
row no two lanes touch one bit (a circulant is a permutation; the staircase bits belong to one check each), so one
barrier a row. The exit on a clear syndrome is uniform across the workgroup, cap 30. Initial L = the int8 LLR through
the bit map and whitening (section 4), R = 0. The 1/2 code's kernel with its stop needs 31,748 B of workgroup memory.

Blocks a frame: the profile's B(V) (2 to 116), in two to four codes. A row is z = 128 lanes of its checks' slots (15
or 16 at 3/4: 19.5 K lane slots an iteration); a failing block runs to the cap or its stop.

### The stop rule (learned)

A codeword that has not cleared is given up by its code's learned stop. A give-up is a stall to everything after it
(verdict 3, ITS -3). A rule can only give up: a block counts only when its syndrome is zero and its CRC passes, so no
rule makes a wrong block verify, and what a rule costs is the blocks it gives up that would clear.
- After each iteration 1 to 29 (the cap ends 30) lane 0
  writes x = (bad, low, first, prev, t, grow) to workgroup memory (the violated checks now, their running minimum,
  after iteration 1, after the iteration before, t, and the posteriors' growth: grow = min(256 S_t / S_1, 65535) in
  integers, S the sum over the n posteriors of min(|L|, 1016), each lane's part added to `magc` as it packs the signs'
  bitmap, 256 where S_1 is 0); lane j < H computes hidden unit j of the first layer, a barrier, of the second and its
  product with the output weight, a barrier, and lane 0 sums the H products in unit order and gives up when the sum is
  under T. No stall test, and nothing of the channel's estimate: est went (a codeword the soft stage under-rates, a
  blurred or mixed capture, was given up after iteration 1; research STATUS "The learned LDPC stop retrained on the
  decoder's own state"). The net's H^2 + 9 H + 1 words (H 32: 1,313 words, 5.3 KB; `stop/rule.mjs` `netWords`, each
  layer input-major so the lanes read consecutive words) sit in TAB's NET, so the stage keeps its 8 storage buffers;
  4 (6 + 2 H) + 4 B of workgroup memory. The nets, their trainer and their files are
  `stop/` (`scripts/gpu/back/stop/README.md`). Each code of the rate profile has its own net, trained the same way
  on that code's neutral channel to its plateau (`stop/stop.safetensors` the 3/4 code's, `stop_78.safetensors`,
  `stop_23.safetensors` and `stop_12.safetensors`, picked by the code's checks, `stop/rule.mjs` `ruleUrlFor`), their
  words one after another in the rule stage's table (`back/tiers.mjs` `ruleContents`).

The question each net answers: given what lane 0 can hold after an iteration, is the codeword hopeless? It is
answered on neutral codewords only (`scripts/gpu/back/stop/neutral.py`: the code through a generic channel and the
soft stage's own demapper, channel information half near the decoding threshold and half from hopeless to clean,
uneven and dead sub-channels, fades the soft stage cannot see apart, gain and erasures; no recorded run informs a
range): codewords to train, as many to set the threshold (val) and as many held out (test), each traced through an
exact integer emulation of the decoder (`scripts/gpu/back/stop/intdec.py`, its magnitude sums exact) with no stop to
the cap; the neutral channel has the capture's physics on top as per-coefficient gains (a linear smear, two pictures
in one exposure, local turns). The net: features bad, low, first, fall = prev - bad (as fractions of the code's m), t
and grow / 256, standardised; two hidden ReLU layers of 32; one logit; clearing codewords' states weighted 1,000 (the threshold sits among them); Adam with a cosine
schedule, trained to a plateau on neutral val. The kernel's form folds the standardisation, 1 / m and fall into the
first layer in float64 (one rounding to float32) and the output bias into T; the threshold is set in that float32
arithmetic by the criterion: the largest T whose lost clearing codewords stay at or under the budget times val's
clearing codewords. The budget is 1 in 1,000.

The arithmetic. Float32 in one order everywhere (the kernel, `stop/rule.mjs` `scoreF32` for the JS mirror,
`scripts/gpu/back/stop/net.py` `score` for the tables); grow is an integer, exact on every device. An H-term sum cannot promise equal decisions on every state
against float64, nor under a compiler that fuses each multiply-add; the flips sit within float32 rounding of T.
`scripts/gpu/back/stop/check_rule.mjs` holds `rule.mjs` to `fit.py`'s words and scores bit for bit.

## 7. CRC-32C, compaction, the result record

On a clear syndrome the workgroup packs the systematic bits (a posterior < 0 is a 1) MSB first into the 120 words of
`packw`. The CRC-32C is split over 119 lanes (`glsl/back_tiers.mjs`, as the sender's paint's `wgsl/back_ldpc.mjs` `CRC_SPLIT`; `back/ref_ldpc.mjs` `crcPowers`):
lane w runs the reflected CRC over its 4-byte chunk from a zero state, multiplies by x^(8 * bytes after it) mod P (a
32-step carry-less multiply against a 120-word table in the code's uniform) and XORs into one workgroup atomic; the
initial state's term and the final inversion are one folded constant. It is compared with bytes 476..479 (word 119) read as a
big endian u32. Equal: verdict 1, `atomicAdd` on REC.count claims a slot, lanes 0..120 write the record's 121 words.
Unequal: verdict 4.
- **V and ITS are one word a block**, u32 verdict and i32 iterations, at `f * blocksMax + b` (WGSL has no byte
  stores). Verdict 0 not run, 1 verified, 2 declined, 3 stall, cap or given up by the stop, 4 CRC; ITS the iterations,
  or the C's codes: -2 declined, -3 stall or given up, -1 cap. The gate zeroes both and REC's count.
- **REC** is u32 count then 484-byte records at 4 + 484 r: word 0 = frame + 65536 block, word 1 = iterations as i32,
  words 2 to 120 the 476 payload bytes. Its cap is B x blocksMax records; a verified block past the
  cap is still counted (BCOUNTS column 4 against column 5).
- **BCOUNTS** columns 3 blocks tried (not declined), 4 verified, 5 records written, 6 sub-channels quantised (8 a
  chunk), 7 iterations run (summed over the frame's codewords).

The host reads REC whole, V, ITS, BCOUNTS and PILOT in the batch's one readback (`core/dec/front.cpp`). The checks
hold each payload after its 4-byte id to the test stream (SHAKE256 of the id: the receiver's judge, the replay's
per-frame `bad`), which must be 0; a receiver has only the CRC, which is what makes the CRC the last word.

## 8. Tables and where each comes from

`M = await init()` (sim/ob.mjs), a `Lizard(n, subch, 1, { span })` built first; the wasm holds one configuration at a
time.

| table | export (call) | shape |
|---|---|---|
| n, bw, vblocks, subch, blocks, npos | `M._lizard_shape(p)`, `Lizard.shape()` | 6 i32 |
| pos, decoded to (u, v) | `M._lizard_pos_ptr()`, `Lizard.posTable()`; v = (p div (n bw)) bw + (p mod (n bw)) mod bw, uw = (p mod (n bw)) div bw, u = uw < n / 2 ? uw : uw - n | subch * 320 u32; entry 0 is (1, 0) |
| cx, qx, X1, X2, sx2, sq2 | `M._lizard_tables_out(p)`, `Lizard.tables()` | 6 n + 2 f32 |
| block first slot, count | `M._lizard_block_runs(pFirst, pCount)`, `Lizard.ldpcCode().first, .count` | blocks i32 each: each block's own first slot of the frame and its slots (4496, 5152, 5760 or 7680 by its code) |
| LDPC dims and schedule | `M._lizard_ldpc_tables(tier, pDims, pLay)` (pLay = 0 sizes it), `Lizard.ldpcCode().code` (tier 0: the profile's innermost code) | dims at 3/4: n 5120, k 3840, m 1280, z 128, zp 128, mb 10, kb 30, norm 13, slotsMax 16, slotsTotal 152, slots 132; lay 275 i32 |
| block bytes | the return of `M._lizard_setup(...)`, `Lizard.blockBytes` | 476 |
| a version's profile | `M._lizard_tiers_for(subch, pTiers)` | up to 4 tiers of (rate, blocks, slots a block) |
| perm, inv, white | not exported; regenerated in JS from section 1 (`bitmap.mjs`) and held to what the C paints by `scripts/gpu/back/check_tiers.mjs` | perm nt; white a bit a slot |
| g0, step | `M._lizard_grid_out(p)` | 2 f32 |
| CRC-32C | `M._ob_test_crc_hash(p, len)` | |
| decline bar, verdicts | `M._ob_test_crc_gate_out(est, blkEst, bar, declined)` | blocks each |
| truth stream | `streamBlock(id, len)` (sim/ob.mjs, `_stream_fill`); `sourceFor(spec.stream)` (sim/phy.mjs) | |
| the C's spectrum | `Lizard.measure(true)` then `read()`: 2 * 320 * subch f32, the detrended coefficients in pos order | |
| the C's verdicts | `Lizard.rxFinish().ok`; `Lizard.blockStats()` its (-2 declined, -3 stall, -1 cap, > 0 iterations) and est | blocks |
| a grid from the C's registration | `M._ob_test_sample(img, iw, ih, 1, 2, dims, state, grid)` | n x n f32 row major, pitch n |
| finish from soft values, verdicts | `Lizard.llrView()` + `rxLdpc(est)` | |
| the exact symbols sent | `Lizard.measure(true)`, `encode(blocks)`, `sent()` | 640 subch int8, +1 / -1 |

Not exported and not needed: the bit map and whitening tables (the JS regeneration is held to the C's paint) and
the C's slot-order LLRs after a finish (`scripts/gpu/back/ref_soft.mjs` gives them from `read()`).
## 9. As composed

`back.mjs` is the chain behind the front half. `BackHalf` owns the transform, the soft and LDPC stages' tables and
the rule stage, and one set of per-frame strides from `dims.mjs derived(sizes, { B })`, which `soft.mjs` and
`ldpc.mjs` also size a lane's buffers from, so no two stages disagree on where frame f's slots sit. The native host
(`core/dec/front.cpp`) makes the lanes and records the chain from the setup `gen/gen.mjs` exports (`exportBack`).

### What the wiring calls

```
const bh = await BackHalf.build(device, { B, sizes, precision, picture, log });   // on gen's recording device
// the native host, per lane: backLane (LISTS, ARGS, PART, Y, S, BCOUNTS, L, BLK, PILOT, EP, V, ITS, REC and the
// bind groups), encodeBack (the six dispatches), and the readback: REC, V, ITS, BCOUNTS, PILOT, bh.readBytes a batch
//   records: { frame, block, its, id, payload (476 B, id first) }, the verified blocks only
//   a frame: verdict, its (blocksMax each), counts (COUNT: columns, rows, stored, tried, verified, records, subch,
//   iterations), pilots (a block's pilot reading), shift (the grid's)
```

- `sizes`: the decoder's eight, at the size index `sel[f].z`, each `{ n, subch }` (the picture at its top) or null for a
  size not built. blocksMax, and with it every stride and the readback, is the largest built size's: 128 8-sub-channel
  chunks with the 1536 picture built (the rate profile's most, 124 blocks, fit under it).
- `picture`: the front half's rings, `[{ span, margin, lattice }]`, from which PIC holds each ring's grid for a picture
  and its lattice.
- `arecBuf`: the front half's anchors records, which pass 1 reads (section 10).
- A size whose pass 1 needs more workgroup memory than the device offers is left unbuilt, and the build logs it. The gate leaves a frame off every list when its size is not built, its `sel[f].z` is
  `SLOTS` or more, its `sel[f].y` is under 2 (not found, or no version from its word or the held one), or its
  `frames[f].valid` is 0.
- The host clears nothing. The gate zeroes BCOUNTS, V, ITS and REC's count at the head of the chain, so the chain can go
  inside an open pass. Every dispatch after the gate is indirect from the gate's ARGS, over the frames it lists (the
  soft values a size's listed frames, the LDPC every listed frame); the gate walks every slot in one thread.
- `BackHalf.build` reads its tables from the wasm (`transformTables`, `ruleTables`). The wasm holds one configuration at
  a time, so build the back half while no other `Lizard` is in use.
- `precision: "f16"` keeps Y and S in f16 where the device has `shader-f16` and falls back to f32 with a log.
- Device limits: workgroup memory at the adapter's maximum. The LDPC needs up to 31,748 B (the 1/2 code with its stop);
  pass 1 at most 18,892 B (f32, the 1536 picture; section 3). Each stage binds at most 8 storage buffers.

### A lane's buffers

Strides are the largest built size's; cbytes is 4 for f16 and 8 for f32.

| buffer | written by, read by | bytes |
|---|---|---|
| frames, sel | the front half's; read only | (not the back half's) |
| LISTS | gate; every stage | 4 (SLOTS (1 + B) + 6 B) |
| ARGS | gate; indirect dispatch of pass 1, reduce, pass 2, the soft stage and the LDPC | 432 |
| PART, COEF | pass 1, reduce; reduce, pass 2 (PART's w is the column mean) | 16 partStride + 32 a frame |
| Y | pass 1; pass 2 | cbytes V n a frame (V - 1 rows written by pass 1) |
| S | pass 2; soft | cbytes 320 subch a frame |
| L, EP, BLK | soft; LDPC | 640 subchMax + 8 subchMax + 8 blocksMax a frame |
| PILOT | soft; host | 4 blocksMax + 8 a frame |
| V, ITS | LDPC; host | 4 blocksMax each a frame |
| REC | LDPC; host | 4 + 484 recCap a batch |
| BCOUNTS | gate (zeroes), transform, rule stage; host | 32 a frame |

The shared tables are built once, not per lane: the transform's twiddles, detrend basis, rows uniform, PIC and the
gate's uniform; the UV and UC table, 4 B an entry each; the rule stage's TAB (every version's row, each code's INV,
WHITE, the learned stops' words) and each code's uniform (its block rows, data slots and the CRC powers).

`bh.readBytes` a batch = (4 + 484 recCap) + 2 (4 B blocksMax) + 32 B + (4 B blocksMax + 8 B) a frame, rounded up to 8.
REC goes over whole because its count exists only on the device; reading count x 484 instead would take a second
round trip.

## 10. The fused pass 1

Pass 1 samples each picture itself: no grid is written or read.

- Each sample: module (g0 + x step, g0 + y step) through `toImage` (`glsl/common.mjs` MAP,
  the lens about the camera's axis included), plus the Coons patch of F7's residuals, read by `bilinH` (`glsl/common.mjs`
  LUMH: one fetch through the texture unit's linear filter, the position clamped to the frame's outer texel centres, so
  a read past the frame takes its edge texel as `bilin0`'s clamp to the frame's own w, h does; the weights the device's
  fixed point), in the same order a thread loads its 8 rows of a column quad. The patch
  is `glsl/sample.mjs` COONS, the one copy of it. A side's curve depends on one coordinate
  only, so the pass takes the top and bottom curves and the weight across once a column and the left and right curves
  and the weight down once a row: 24 curves a thread for its 32 samples, where a sample at a time evaluates 128.
- Bindings: 8 storage buffers (LISTS, twiddles, PART, Y, BCOUNTS, maps, resid, and the anchors' records), the frame
  texture and the linear sampler its bilinear reads go through (binding 11: one filtered fetch a sample, the position
  clamped to the frame's outer texel centres), and two uniforms:
  the params and PIC (a size's grid in each ring, `gpu/tables.mjs` gridOf, and the rings' lattices). The gate writes
  every frame's (w, h) and ring into LISTS, and the pass reads the edge clamp and the ring from there.
- `sel[f]` is written by PICK (F6) in find mode, cleared at the head of every find batch, or by the host when it hands
  the maps over; F8 sets its version and picture. The pass samples exactly the frames the gate lists. BCOUNTS column 0
  (columns transformed, n a listed frame) is the check that every column was sampled.
- The detrend needs nothing more: the partials come from pass 1's load of the fused samples; reduce and pass 2's
  spectral subtraction read PART and COEF.
- The anchors (SPEC 6.13): pass 1 reads the front half's record a frame (binding 10, written by F7a's ANCH_FIT),
  adds the knots' tensor cubic Hermite mesh (the C's `anch_herm4`) to every sample's position before its bilinear
  read, and sets every sample within the square's half side plus 2.5 of the picture's samples of an anchor's centre,
  on both axes, to the picture's mean before z and the detrend's sums, the erased ranges decided in integers on pass
  1's own grid (no device's rounding decides a sample).
