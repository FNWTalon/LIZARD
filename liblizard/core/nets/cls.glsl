// ai: cnn-v3, the cascade's small classifier, int8, on the cooperative-matrix units (liblizard/gen/nets_cls.mjs
// ai: packs the weights and compiles this with the net's -D offsets). The kernel replaces the
// ai: WGSL twin (liblizard/gpu/wgsl/classify_gemm.mjs, int8 form): the same bindings, the same dispatch (cap workgroups a
// ai: frame, y the frame), the same outputs to the bit (each slot under the frame's count: its score and its reading's
// ai: twelve words; nothing past the count), the contract's codes (liblizard/gpu/cnn/q8c.mjs forwardCodesC) on the
// ai: kernel's own Xq. On the Adreno 840 the fast build costs 0.36 ms a frame of 1,024 patches against the twin's 0.48
// ai: (2026-09-30, int8-sg, 96 recorded frames; core/nets/README.md).
// ai:
// ai: A workgroup is 64 invocations and takes PP patches (4 by default: workgroup x takes x 4 .. x 4 + 3; the rest of
// ai: the dispatch returns at once). Each patch's cut, statistics and Xq are the WGSL's own arithmetic in its own order
// ai: (64 lanes a patch, each lane the WGSL lane's four runs of four samples; the sums reduced in the WGSL's order), so
// ai: Xq is the WGSL's. Every conv is then a GEMM of 64-row blocks: a row is one output position (conv1's: four), its
// ai: K the im2col bytes of those outputs in 32-byte steps, B the packed weights; the block's C is requantised to
// ai: codes, clamp(roundEven(f32(acc + bq) m), 0, 127), and written as the next layer's NHWC map, so the next layer's
// ai: rows are copies of its words. fc1 is the transposed GEMM split in K (A the weights, a row an output over half of
// ai: K; B the last conv's maps, a column a patch's half, strided in shared memory; the two diagonal blocks summed),
// ai: relu(f32(acc + bq) s); fc2 and the readings are the WGSL's, in f32 in its order.
// ai:
// ai: Two ways a block reaches the matrix units, the same arithmetic, one a build (FAST is fixed at specialisation:
// ai: with both arms in one shader the Adreno runs the fast one twelve times slower):
// ai:   fast     (FAST 1, M 64, NW 32; the Adreno): lane l builds GEMM row l in registers and writes it into the A
// ai:            matrix's elements, and reads C row l's elements back, where the device's element layout is lane = row,
// ai:            element i = column i and its stride unit is SU. The kernel checks both on every workgroup (tables
// ai:            loaded through coopMatLoad); a workgroup where they do not hold writes its patches as no mark.
// ai:   staging  (FAST 0, any M): the rows through shared memory, coopMatLoad per M x 32 tile, the accumulator through
// ai:            shared memory per K step (a subgroup a tile), so any M, N tile, subgroup size and stride unit works
// ai:            (the 4090's 16 x 16 x 32 checks it on the desktop).
// ai:
// ai: The stride unit. SPIR-V counts coopMatLoad's stride in elements of the array loaded from (u32 words here); the
// ai: Adreno 840's driver counts it in the matrix's components (bytes for an int8 matrix). The fast build's strides are
// ai: in the unit -DSU names (1 words, 4 bytes) and its layout check verifies it; the staging build's take the unit
// ai: a probe finds on the device (strideUnit).
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
// ai: M: the coopmat's rows; NW: the wide N tile (32 where the device has 64 x 32 x 32, else 16); FAST: 1 the fast
// ai: build, 0 the staging build; STOP: a profiling build ends the net after layer STOP (1 the cut, 2 conv1, 3 conv2,
// ai: 4 conv3, which the Adreno does not compile in the fast build) and writes readings from zeros; DBG 0 in a shipped
// ai: spec (the readings then equal the WGSL twin's); DBG 1 (a check build): each reading's second vec4 carries the
// ai: path its workgroup took in y (1 fast, 2 staging, 3 a fast build whose check failed) and the stride unit in z,
// ai: where the WGSL writes 0;
// ai: PP: patches a workgroup (1 to 4; conv3's rows past PP read allocated words and are not used). -DSU (a build
// ai: define, not a specialisation constant): the stride unit the fast path is built for (1u or 4u).
layout(constant_id = 0) const uint M = 64u;
layout(constant_id = 1) const uint NW = 32u;
layout(constant_id = 2) const uint FAST = 1u;
layout(constant_id = 3) const uint STOP = 0u;
layout(constant_id = 4) const uint DBG = 0u;
layout(constant_id = 5) const uint PP = 4u;
const bool FASTOK = FAST == 1u && M == 64u && NW == 32u;

#define RM gl_CooperativeMatrixLayoutRowMajor
#define CM gl_CooperativeMatrixLayoutColumnMajor
#define A_t coopmat<int8_t, gl_ScopeSubgroup, M, 32, gl_MatrixUseA>
#define B16_t coopmat<int8_t, gl_ScopeSubgroup, 32, 16, gl_MatrixUseB>
#define BW_t coopmat<int8_t, gl_ScopeSubgroup, 32, NW, gl_MatrixUseB>
#define C16_t coopmat<int32_t, gl_ScopeSubgroup, M, 16, gl_MatrixUseAccumulator>
#define CW_t coopmat<int32_t, gl_ScopeSubgroup, M, NW, gl_MatrixUseAccumulator>

struct Frame { uint w, h, size, valid; };
layout(set = 0, binding = 0) uniform texture2DArray img;
layout(set = 0, binding = 1, std430) readonly buffer Lv1 { float lv1[]; };
layout(set = 0, binding = 2, std430) readonly buffer Lv2 { float lv2[]; };
layout(set = 0, binding = 3, std430) readonly buffer Lv3 { float lv3[]; };
layout(set = 0, binding = 4, std430) readonly buffer Lv4 { float lv4[]; };
layout(set = 0, binding = 5, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 6, std140) uniform LevelDims { uvec4 d[5]; } LD;
layout(set = 0, binding = 7, std430) readonly buffer Peaks { vec4 peaks[]; };
layout(set = 0, binding = 8, std430) readonly buffer Counts { uint counts[]; };
layout(set = 0, binding = 9, std430) buffer Readings { vec4 readings[]; };
layout(set = 0, binding = 10, std140) uniform P4 { uint cap, p0, p1, p2; } P;
layout(set = 0, binding = 11, std430) readonly buffer Weights { uint ws[]; };
layout(set = 0, binding = 12, std430) buffer Score { float score[]; };

const float KAPPA = 1.8;
const float PI = 3.14159265;
const float QMAX = 127.0;

// ai: Shared memory, u32 words (every region 16-byte aligned): RED the statistics, then fc1's and fc2's outputs (f32
// ai: bits); X a patch's Xq ([y][x], four samples a word); Y its conv1 map (NHWC); Z the four patches' conv2 maps; W
// ai: (over X and Y once the patches are done) their conv3 maps, fc1's B; ST the staging path's A rows (SA, 64 x 8
// ai: words) and accumulator (SC, 64 x 32), and at the start the stride probe's and the layout check's tables. fc1
// ai: reads 16 columns of B, 4 of them patches: the other 12 read the words after W, which are in the array.
const uint RED = 0u, X = 160u, Y = 416u, Z = 928u, W = X, ST = 1952u;
const uint SA = ST, SC = ST + 512u, SMN = ST + 2560u;
shared uint sm[SMN];

float redf(uint i) { return uintBitsToFloat(sm[RED + i]); }
void setRedf(uint i, float v) { sm[RED + i] = floatBitsToUint(v); }

uint levelFor(float u) { return uint(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0)); }

// ai: ---- The cut: classify_gemm.mjs cutAt, a level's loop (the level's size, base and fetch fixed outside it) ----
#define F_TEX(x, y, row) texelFetch(img, ivec3(x, y, int(f)), 0).r
#define F_LV1(x, y, row) lv1[row + uint(x)]
#define F_LV2(x, y, row) lv2[row + uint(x)]
#define F_LV3(x, y, row) lv3[row + uint(x)]
#define F_LV4(x, y, row) lv4[row + uint(x)]
#define DEF_CUT(NAME, FETCH) \
void NAME(vec4 pk, float u, float inv, int lw, int lh, uint lbase, uint lsx, uint f, uint li, inout vec4 sv[4], inout float ssum, inout float ssq) { \
  [[unroll]] for (uint k = 0u; k < 4u; k++) { \
    uint q = li + 64u * k; \
    float sy = float(q / 8u); \
    float sx = float((q % 8u) * 4u); \
    float py = (pk.y + u * (sy - 15.5)) * inv - 0.5; \
    float fy = floor(py); \
    float ay = py - fy; \
    int jy = int(fy); \
    int iy0 = clamp(jy, 0, lh - 1); \
    int iy1 = clamp(jy + 1, 0, lh - 1); \
    uint row0 = lbase + uint(iy0) * lsx; \
    uint row1 = lbase + uint(iy1) * lsx; \
    vec4 s = vec4(0.0); \
    [[unroll]] for (uint t = 0u; t < 4u; t++) { \
      float px = (pk.x + u * ((sx + float(t)) - 15.5)) * inv - 0.5; \
      float fx = floor(px); \
      float ax = px - fx; \
      int jx = int(fx); \
      int ix0 = clamp(jx, 0, lw - 1); \
      int ix1 = clamp(jx + 1, 0, lw - 1); \
      s[t] = mix(mix(FETCH(ix0, iy0, row0), FETCH(ix1, iy0, row0), ax), mix(FETCH(ix0, iy1, row1), FETCH(ix1, iy1, row1), ax), ay); \
    } \
    sv[k] = s; \
    ssum += s.x + s.y + s.z + s.w; \
    ssq += dot(s, s); \
  } \
}
DEF_CUT(cut0, F_TEX)
DEF_CUT(cut1, F_LV1)
DEF_CUT(cut2, F_LV2)
DEF_CUT(cut3, F_LV3)
DEF_CUT(cut4, F_LV4)

// ai: One patch (slot jc of the frame, or slot 0 past its count, as the WGSL) cut, standardised and quantised into X.
// ai: The sums are the WGSL's: a lane's own sixteen samples in run order, eight lanes' in lane order, eight partials in
// ai: order.
void cutPatch(uint f, uint n, uint jc, uint li) {
  uint ic = jc < n ? jc : 0u;
  vec4 pk = peaks[f * P.cap + ic];
  float u = pk.z / KAPPA;
  uint l = levelFor(u);
  Frame F = frames[f];
  int lw = int((F.w + (1u << l) - 1u) >> l);
  int lh = int((F.h + (1u << l) - 1u) >> l);
  uvec4 ls = LD.d[l];
  uint lsx = ls.x;
  uint lbase = f * ls.y * lsx;
  float inv = 1.0 / float(1u << l);
  vec4 sv[4];
  float ssum = 0.0, ssq = 0.0;
  switch (l) {
    case 0u: cut0(pk, u, inv, lw, lh, lbase, lsx, f, li, sv, ssum, ssq); break;
    case 1u: cut1(pk, u, inv, lw, lh, lbase, lsx, f, li, sv, ssum, ssq); break;
    case 2u: cut2(pk, u, inv, lw, lh, lbase, lsx, f, li, sv, ssum, ssq); break;
    case 3u: cut3(pk, u, inv, lw, lh, lbase, lsx, f, li, sv, ssum, ssq); break;
    default: cut4(pk, u, inv, lw, lh, lbase, lsx, f, li, sv, ssum, ssq); break;
  }
  setRedf(li, ssum);
  setRedf(64u + li, ssq);
  barrier();
  if (li < 8u) {
    float t1 = 0.0, t2 = 0.0;
    for (uint k = 0u; k < 8u; k++) { t1 += redf(li * 8u + k); t2 += redf(64u + li * 8u + k); }
    setRedf(128u + li, t1);
    setRedf(136u + li, t2);
  }
  barrier();
  float tsum = 0.0, tsq = 0.0;
  for (uint k = 0u; k < 8u; k++) { tsum += redf(128u + k); tsq += redf(136u + k); }
  float mu = tsum / 1024.0;
  float sd = sqrt(max(tsq / 1024.0 - mu * mu, 0.0));
  [[unroll]] for (uint k = 0u; k < 4u; k++) {
    ivec4 xi = ivec4(clamp(roundEven(((sv[k] - vec4(mu)) / (sd + 0.02)) * INV), vec4(-QMAX), vec4(QMAX)));
    sm[X + li + 64u * k] = (uint(xi.x) & 255u) | ((uint(xi.y) & 255u) << 8u) | ((uint(xi.z) & 255u) << 16u) | (uint(xi.w) << 24u);
  }
}

// ai: ---- GEMM rows: the 32 bytes (8 words) row r of a layer takes at K step s, from its input map ----

// ai: conv1 (c1): r = 4 y + xg, outputs (y, 4 xg + j); words 2 ky, 2 ky + 1 the input row 2 y - 1 + ky at columns
// ai: 8 xg .. 8 xg + 7, word 6 those rows' column 8 xg - 1 (byte ky), word 7 zero.
void rowConv1(uint r, out uint u[8]) {
  uint y = r >> 2u, xg = r & 3u, L = 0u;
  [[unroll]] for (uint ky = 0u; ky < 3u; ky++) {
    int iy = int(2u * y + ky) - 1;
    bool ok = iy >= 0 && iy < 32;
    uint b = X + uint(clamp(iy, 0, 31)) * 8u + 2u * xg;
    u[2u * ky] = ok ? sm[b] : 0u;
    u[2u * ky + 1u] = ok ? sm[b + 1u] : 0u;
    uint lb = sm[max(b, X + 1u) - 1u] >> 24u;
    L |= (ok && xg > 0u ? lb : 0u) << (8u * ky);
  }
  u[6] = L;
  u[7] = 0u;
}

// ai: conv2 (s2c8): r = 8 y + x of 8 x 8; step ky: the conv1 map's row 2 y - 1 + ky, positions 2 x - 1 .. 2 x + 1
// ai: (words 4 x - 2 .. 4 x + 3 of its 32), then two zero words.
void rowConv2(uint r, uint ky, out uint u[8]) {
  uint y = r >> 3u, x = r & 7u;
  int iy = int(2u * y + ky) - 1;
  bool ok = iy >= 0 && iy < 16;
  uint b = Y + uint(clamp(iy, 0, 15)) * 32u;
  [[unroll]] for (uint j = 0u; j < 6u; j++) {
    int w = int(4u * x + j) - 2;
    u[j] = ok && w >= 0 ? sm[b + uint(max(w, 0))] : 0u;
  }
  u[6] = 0u;
  u[7] = 0u;
}
// ai: conv3 (s2t2): r = 16 p + 4 y + x (the four patches' 4 x 4 in one block); step s: taps 2 s, 2 s + 1 of patch p's
// ai: conv2 map (8 x 8 x 16, 4 words a position).
void rowConv3(uint r, uint s, out uint u[8]) {
  uint p = r >> 4u, y = (r >> 2u) & 3u, x = r & 3u;
  [[unroll]] for (uint h = 0u; h < 2u; h++) {
    uint t = 2u * s + h;
    int iy = int(2u * y + t / 3u) - 1, ix = int(2u * x + t % 3u) - 1;
    bool ok = t < 9u && iy >= 0 && iy < 8 && ix >= 0 && ix < 8;
    uint b = Z + p * 256u + uint(clamp(iy, 0, 7) * 8 + clamp(ix, 0, 7)) * 4u;
    [[unroll]] for (uint j = 0u; j < 4u; j++) u[4u * h + j] = ok ? sm[b + j] : 0u;
  }
}

// ai: A row's bytes into the A matrix's elements (the fast path: element i is byte i of the lane's row).
#define ROW_TO_A(a, u) [[unroll]] for (uint i_ = 0u; i_ < 32u; i_++) a[i_] = int8_t(bitfieldExtract(int(u[i_ >> 2u]), int(8u * (i_ & 3u)), 8));

// ai: A C row's NC accumulators requantised to codes (column c's bias and scale at OBQ + c, OM + c) and packed four a
// ai: word into dst .. dst + NC / 4.
#define REQUANT_ROW(vals, NC, OBQ, OM, dst) { \
  [[unroll]] for (uint w_ = 0u; w_ < NC / 4u; w_++) { \
    uint o_ = 0u; \
    [[unroll]] for (uint e_ = 0u; e_ < 4u; e_++) { \
      uint c_ = 4u * w_ + e_; \
      float v_ = clamp(roundEven(float(vals[c_] + int(ws[OBQ + c_])) * uintBitsToFloat(ws[OM + c_])), 0.0, QMAX); \
      o_ |= uint(v_) << (8u * e_); \
    } \
    sm[dst + w_] = o_; \
  } }

// ai: One 64-row block of a conv: S K steps, NC columns (16 or 32), B at OB (step s at OB + 8 NC s words), rows by
// ai: ROWF(r, s, u), row r = RB + lane; the codes of row r to DSTF(r). BT/CT: the matrix types of NC's width. The
// ai: staging path's N tile for a 32-wide layer is NW (16 on the 4090); a 16-wide layer's is 16 (NW 32 and NC 16 do
// ai: not mix: 16-wide layers take the 16-wide types, where NTL is 1).
#define CONV_BLOCK(S, NC, BT, CT, OB, OBQ, OM, RB, ROWF, DSTF) { \
  int vals_[NC]; \
  uint r_ = (RB) + li; \
  if (FASTOK) { \
    CT acc_ = CT(0); \
    [[unroll]] for (uint s_ = 0u; s_ < (S); s_++) { \
      uint u_[8]; ROWF(r_, s_, u_); \
      A_t a_; ROW_TO_A(a_, u_) \
      BT b_; coopMatLoad(b_, ws, (OB) + 8u * (NC) * s_, ((NC) / 4u) * SU, RM); \
      acc_ = coopMatMulAdd(a_, b_, acc_); \
    } \
    [[unroll]] for (uint c_ = 0u; c_ < (NC); c_++) vals_[c_] = acc_[c_]; \
  } else { \
    [[unroll]] for (uint c_ = 0u; c_ < (NC); c_++) sm[SC + (NC) * li + c_] = 0u; \
    for (uint s_ = 0u; s_ < (S); s_++) { \
      uint u_[8]; ROWF(r_, s_, u_); \
      [[unroll]] for (uint j_ = 0u; j_ < 8u; j_++) sm[SA + 8u * li + j_] = u_[j_]; \
      barrier(); \
      const uint NTL_ = (NC) / NW < 1u ? 1u : (NC) / NW; \
      for (uint t_ = gl_SubgroupID; t_ < (64u / M) * NTL_; t_ += gl_NumSubgroups) { \
        uint mt_ = t_ / NTL_, nt_ = t_ % NTL_; \
        A_t a_; coopMatLoad(a_, sm, SA + mt_ * M * 8u, 8u * su, RM); \
        BT b_; coopMatLoad(b_, ws, (OB) + 8u * (NC) * s_ + nt_ * (NW / 4u), ((NC) / 4u) * su, RM); \
        CT c_; coopMatLoad(c_, sm, SC + mt_ * M * (NC) + nt_ * NW, (NC), RM); \
        c_ = coopMatMulAdd(a_, b_, c_); \
        coopMatStore(c_, sm, SC + mt_ * M * (NC) + nt_ * NW, (NC), RM); \
      } \
      barrier(); \
    } \
    [[unroll]] for (uint c_ = 0u; c_ < (NC); c_++) vals_[c_] = int(sm[SC + (NC) * li + c_]); \
  } \
  REQUANT_ROW(vals_, (NC), (OBQ), (OM), DSTF(r_)) \
  barrier(); }

// ai: ---- The stride unit: what coopMatLoad's stride counts for an int8 matrix in this u32 memory, on this device.
// ai: Found from a store, which no element layout enters: a matrix whose rows are bytes 0..31 stored with stride 8
// ai: puts its last row's last byte at byte 32 M - 1 where a stride is words, and ends before it where it is bytes.
// ai: Returns the factor a stride in words takes for an int8 matrix: 1 or 4. int32 matrices are words either way. ----
uint strideUnit(uint li) {
  const uint T = ST + 512u;
  if (li < 8u) sm[T + li] = (4u * li) | ((4u * li + 1u) << 8u) | ((4u * li + 2u) << 16u) | ((4u * li + 3u) << 24u);
  if (li == 0u) sm[ST + 8u * M - 1u] = 0u;
  barrier();
  A_t a;
  coopMatLoad(a, sm, T, 0u, RM);
  coopMatStore(a, sm, ST, 8u, RM);
  barrier();
  uint su = (sm[ST + 8u * M - 1u] >> 24u) == 31u ? 1u : 4u;
  barrier();
  return su;
}

// ai: ---- The layout check (the fast path's premise): A's element i of lane l is (l, i), C's likewise, and a stride
// ai: counts SU ----
bool fastLayout(uint li) {
  // ai: T0: bytes 0..31 = 0..31 (A with stride 0: every row 0..31); T1: byte b = b >> 4 (A with a 16-byte stride: row r
  // ai: element k = r + (k >> 4)); T3: word u = u (C with stride 0: row c = c; with stride 4: row r element c = 4 r + c).
  const uint T0 = ST, T1 = ST + 8u, T3 = ST + 268u;
  if (li < 8u) sm[T0 + li] = (4u * li) | ((4u * li + 1u) << 8u) | ((4u * li + 2u) << 16u) | ((4u * li + 3u) << 24u);
  for (uint i = li; i < 260u; i += 64u) sm[T1 + i] = (i >> 2u) * 0x01010101u;
  for (uint i = li; i < 284u; i += 64u) sm[T3 + i] = i;
  if (li == 0u) sm[RED + 159u] = 0u;
  barrier();
  A_t a0, a1;
  coopMatLoad(a0, sm, T0, 0u, RM);
  coopMatLoad(a1, sm, T1, 4u * SU, RM);
  C16_t c0, c1;
  coopMatLoad(c0, sm, T3, 0u, RM);
  coopMatLoad(c1, sm, T3, 4u, RM);
  CW_t d0, d1;
  coopMatLoad(d0, sm, T3, 0u, RM);
  coopMatLoad(d1, sm, T3, 4u, RM);
  bool ok = a0.length() == 32 && c0.length() == 16 && d0.length() == 32;
  if (ok) {
    [[unroll]] for (uint i = 0u; i < 32u; i++) ok = ok && int(a0[i]) == int(i) && int(a1[i]) == int(li + (i >> 4u));
    [[unroll]] for (uint i = 0u; i < 16u; i++) ok = ok && c0[i] == int(i) && c1[i] == int(4u * li + i);
    [[unroll]] for (uint i = 0u; i < 32u; i++) ok = ok && d0[i] == int(i) && d1[i] == int(4u * li + i);
  }
  if (!ok) atomicOr(sm[RED + 159u], 1u);
  barrier();
  bool all = sm[RED + 159u] == 0u;
  barrier();
  return all;
}

// ai: ---- fc1, transposed and split in K: C = A B over K / 2, A the weights (row h 32 + o: output o over K's half h),
// ai: B column 2 n + h' patch n's half h' of its last-conv map (MAPF + (K1 / 8) (2 n + h') words: the maps are K1
// ai: bytes apart), so patch n's output o is C[o][2 n] + C[32 + o][2 n + 1]; relu(f32(acc + bq) s) into RED n 32 + o.
// ai: Columns past patch 3 read the words after the maps (in the array) and are not used. ----
void fc1(uint su, uint li, uint MAPF) {
  int part[PP];
  uint h = li >> 5u;
  if (FASTOK) {
    CW_t acc = CW_t(0);
    for (uint s = 0u; s < SF1; s++) {
      A_t a; coopMatLoad(a, ws, OAF1 + 8u * s, (K1 / 8u) * SU, RM);
      BW_t b; coopMatLoad(b, sm, MAPF + 8u * s, (K1 / 8u) * SU, CM);
      acc = coopMatMulAdd(a, b, acc);
    }
    [[unroll]] for (uint c = 0u; c < PP; c++) { int e0 = acc[2u * c], e1 = acc[2u * c + 1u]; part[c] = h == 0u ? e0 : e1; }
  } else {
    [[unroll]] for (uint c = 0u; c < 32u; c++) sm[SC + 32u * li + c] = 0u;
    barrier();
    const uint NTL = 32u / NW;
    for (uint s = 0u; s < SF1; s++) {
      for (uint t = gl_SubgroupID; t < (64u / M) * NTL; t += gl_NumSubgroups) {
        uint mt = t / NTL, nt = t % NTL;
        A_t a; coopMatLoad(a, ws, OAF1 + mt * M * (K1 / 8u) + 8u * s, (K1 / 8u) * su, RM);
        BW_t b; coopMatLoad(b, sm, MAPF + nt * NW * (K1 / 8u) + 8u * s, (K1 / 8u) * su, CM);
        CW_t c; coopMatLoad(c, sm, SC + mt * M * 32u + nt * NW, 32u, RM);
        c = coopMatMulAdd(a, b, c);
        coopMatStore(c, sm, SC + mt * M * 32u + nt * NW, 32u, RM);
      }
      barrier();
    }
    [[unroll]] for (uint c = 0u; c < PP; c++) part[c] = int(sm[SC + 32u * li + 2u * c + h]);
  }
  barrier();
  if (h == 1u) [[unroll]] for (uint c = 0u; c < PP; c++) sm[ST + (li - 32u) * PP + c] = uint(part[c]);
  barrier();
  if (h == 0u) {
    float s1 = uintBitsToFloat(ws[OSF1 + li]);
    int bq = int(ws[OBQF1 + li]);
    [[unroll]] for (uint c = 0u; c < PP; c++) setRedf(c * 32u + li, max(float(part[c] + int(sm[ST + li * PP + c]) + bq) * s1, 0.0));
  }
  barrier();
}

// ai: fc2 and the readings, the WGSL's: 20 lanes (patch, output) sum in f32 in its order, then a lane a patch.
// ai: nomark (a fast build whose layout check failed): the patches read as no mark (a logit of -1e30), so the frame
// ai: reads nothing where the kernel cannot compute it.
void finish(uint f, uint n, uint j0, uint li, bool zeros, bool nomark, uint su) {
  if (li < 5u * PP) {
    uint p = li / 5u, o = li % 5u;
    float acc = uintBitsToFloat(ws[OBF2 + o]);
    for (uint v = 0u; v < 8u; v++) {
      uint k = v * 4u;
      [[unroll]] for (uint e = 0u; e < 4u; e++) acc += uintBitsToFloat(ws[OWF2 + o * 32u + k + e]) * redf(p * 32u + k + e);
    }
    setRedf(128u + li, nomark ? (o == 0u ? -1e30 : o == 1u ? 1.0 : 0.0) : zeros ? 0.0 : acc);
  }
  barrier();
  if (li < PP) {
    uint j = j0 + li;
    if (j < n) {
      vec4 pw = peaks[f * P.cap + j];
      float uw = pw.z / KAPPA;
      float z0 = redf(128u + li * 5u);
      float z1 = redf(128u + li * 5u + 1u);
      float z2 = redf(128u + li * 5u + 2u);
      float z3 = redf(128u + li * 5u + 3u);
      float z4 = redf(128u + li * 5u + 4u);
      float t = atan(z2, z1);
      float pr = 1.0 / (1.0 + exp(-z0));
      uint o3 = 3u * (f * P.cap + j);
      score[f * P.cap + j] = z0;
      readings[o3] = vec4(pw.xy, uw * exp(z3), t - PI / 4.0);
      readings[o3 + 1u] = vec4(pr, DBG == 1u ? (nomark ? 3.0 : FASTOK ? 1.0 : 2.0) : 0.0, DBG == 1u ? float(su) : 0.0, 0.0);
      readings[o3 + 2u] = vec4(z4 > 0.0 ? 1.0 : 0.0, pw.w, float(levelFor(uw)), 1.0);
    }
  }
}

// ai: Each conv's rows and where row r's codes go (words): conv1 the patch's conv1 map; conv2 patch cp's conv2 map in
// ai: Z; conv3 the four patches' conv3 maps (W).
#define ROW1(r, s, u) rowConv1(r, u)
#define DST1(r) (Y + 8u * (r))
#define DST2(r) (Z + cp * 256u + 4u * (r))
#define DST3(r) (W + 4u * (r))

void main() {
  uint li = gl_LocalInvocationIndex;
  uint f = gl_WorkGroupID.y;
  // ai: uniform over the workgroup (its id and read-only storage), so the group returns before any barrier
  if (frames[f].valid == 0u) return;
  uint n = min(counts[f * 16u + 2u], P.cap);
  uint j0 = gl_WorkGroupID.x * PP;
  if (j0 >= n) return;
  // ai: one arm a build (FASTOK is fixed at specialisation): the fast build checks its premise and, where it fails,
  // ai: writes its patches as no mark; the staging build probes the device's stride unit
  uint su = SU;
  bool ok = true;
  if (FASTOK) ok = fastLayout(li);
  else su = strideUnit(li);
  if (!ok) {
    finish(f, n, j0, li, true, true, su);
    return;
  }
  for (uint cp = 0u; cp < PP; cp++) {
    cutPatch(f, n, j0 + cp, li);
    barrier();
    if (STOP == 1u) continue;
    CONV_BLOCK(1u, 32u, BW_t, CW_t, OB1, OBQ1, OM1, 0u, ROW1, DST1)
    if (STOP == 2u) continue;
    CONV_BLOCK(3u, 16u, B16_t, C16_t, OB2, OBQ2, OM2, 0u, rowConv2, DST2)
  }
  bool zeros = STOP != 0u;
  if (STOP == 0u || STOP > 3u) {
    CONV_BLOCK(5u, 16u, B16_t, C16_t, OB3, OBQ3, OM3, 0u, rowConv3, DST3)
    if (STOP == 0u || STOP > 4u) fc1(su, li, W);
  }
  finish(f, n, j0, li, zeros, false, su);
}
