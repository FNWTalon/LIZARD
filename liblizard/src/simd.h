// ai: The codec's vector paths are written once, against wasm's SIMD intrinsics (wasm_simd128.h), each beside a
// ai: scalar twin doing the same operations in the same order. This header is what they are compiled against:
// ai:   wasm with -msimd128      <wasm_simd128.h> itself
// ai:   arm64 with NEON          v128_t and every wasm intrinsic the source uses, each over the NEON instruction of
// ai:                            the same arithmetic (little-endian; -DOB_SCALAR leaves it out and builds the twins)
// ai:   anything else            nothing: the scalar twins
// ai: OB_SIMD is defined where the vector paths exist, and is what the source tests.
// ai: A NEON wrapper is wasm's operation lane for lane, the cases a scalar twin can tell apart included (NaN, a
// ai: saturating conversion, a shift count, an out-of-range shuffle index): test/neon_check.c holds each wrapper to
// ai: a plain C model of the wasm instruction, and the same model to wasm's own intrinsics under emcc.
// ai: Every build line carries -ffp-contract=off, here as on wasm: NEON has a fused multiply-add and a compiler
// ai: allowed to contract would round once where the source rounds twice.
#ifndef OB_SIMD_H
#define OB_SIMD_H

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define OB_SIMD 1

#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && !defined(OB_SCALAR)
#include <arm_neon.h>
#include <stdint.h>
#define OB_SIMD 1

// ai: One type for every lane view, as wasm's v128_t is: four int32 lanes carry the bits and each wrapper
// ai: reinterprets them as the lanes its operation works on.
typedef int32x4_t v128_t;
#define OB_VFN static inline __attribute__((__always_inline__))
#define OB_F32(a) vreinterpretq_f32_s32(a)
#define OB_S16(a) vreinterpretq_s16_s32(a)
#define OB_U16(a) vreinterpretq_u16_s32(a)
#define OB_S8(a) vreinterpretq_s8_s32(a)
#define OB_U8(a) vreinterpretq_u8_s32(a)
#define OB_U32(a) vreinterpretq_u32_s32(a)
#define OB_OF_F32(a) vreinterpretq_s32_f32(a)
#define OB_OF_S16(a) vreinterpretq_s32_s16(a)
#define OB_OF_U16(a) vreinterpretq_s32_u16(a)
#define OB_OF_S8(a) vreinterpretq_s32_s8(a)
#define OB_OF_U8(a) vreinterpretq_s32_u8(a)
#define OB_OF_U32(a) vreinterpretq_s32_u32(a)

// ai: Memory. Sixteen bytes at any alignment and of any type, read and written as bytes.
OB_VFN v128_t wasm_v128_load(const void *p) { return OB_OF_U8(vld1q_u8((const uint8_t *)p)); }
OB_VFN void wasm_v128_store(void *p, v128_t a) { vst1q_u8((uint8_t *)p, OB_U8(a)); }
// ai: Eight signed bytes, each widened to a 16-bit lane.
OB_VFN v128_t wasm_i16x8_load8x8(const void *p) { return OB_OF_S16(vmovl_s8(vld1_s8((const int8_t *)p))); }

// ai: Bits.
OB_VFN v128_t wasm_v128_and(v128_t a, v128_t b) { return vandq_s32(a, b); }
OB_VFN v128_t wasm_v128_or(v128_t a, v128_t b) { return vorrq_s32(a, b); }
OB_VFN v128_t wasm_v128_xor(v128_t a, v128_t b) { return veorq_s32(a, b); }
// ai: (a & mask) | (b & ~mask), bit by bit: BSL with the mask as its first operand.
OB_VFN v128_t wasm_v128_bitselect(v128_t a, v128_t b, v128_t mask) { return vbslq_s32(OB_U32(mask), a, b); }

// ai: f32x4. Add, subtract, multiply and divide are IEEE single, rounded to nearest even, one rounding each.
OB_VFN v128_t wasm_f32x4_splat(float a) { return OB_OF_F32(vdupq_n_f32(a)); }
OB_VFN v128_t wasm_f32x4_make(float c0, float c1, float c2, float c3) { const float t[4] = { c0, c1, c2, c3 }; return OB_OF_F32(vld1q_f32(t)); }
OB_VFN v128_t wasm_f32x4_add(v128_t a, v128_t b) { return OB_OF_F32(vaddq_f32(OB_F32(a), OB_F32(b))); }
OB_VFN v128_t wasm_f32x4_sub(v128_t a, v128_t b) { return OB_OF_F32(vsubq_f32(OB_F32(a), OB_F32(b))); }
OB_VFN v128_t wasm_f32x4_mul(v128_t a, v128_t b) { return OB_OF_F32(vmulq_f32(OB_F32(a), OB_F32(b))); }
OB_VFN v128_t wasm_f32x4_div(v128_t a, v128_t b) { return OB_OF_F32(vdivq_f32(OB_F32(a), OB_F32(b))); }
OB_VFN v128_t wasm_f32x4_abs(v128_t a) { return OB_OF_F32(vabsq_f32(OB_F32(a))); }
// ai: To the nearest whole number, halves to the even one (FRINTN), whatever the rounding mode.
OB_VFN v128_t wasm_f32x4_nearest(v128_t a) { return OB_OF_F32(vrndnq_f32(OB_F32(a))); }
// ai: pmin(a, b) is b < a ? b : a and pmax(a, b) is a < b ? b : a: a select on one compare, so a NaN in either
// ai: operand gives a back, and of two zeros a. NEON's own minimum and maximum (FMIN, FMINNM) answer both
// ai: differently, so they are not used.
OB_VFN v128_t wasm_f32x4_pmin(v128_t a, v128_t b) { return OB_OF_F32(vbslq_f32(vcltq_f32(OB_F32(b), OB_F32(a)), OB_F32(b), OB_F32(a))); }
OB_VFN v128_t wasm_f32x4_pmax(v128_t a, v128_t b) { return OB_OF_F32(vbslq_f32(vcltq_f32(OB_F32(a), OB_F32(b)), OB_F32(b), OB_F32(a))); }
// ai: Compares: a lane of all ones where true, zeros where false, and false whenever either side is NaN.
OB_VFN v128_t wasm_f32x4_lt(v128_t a, v128_t b) { return OB_OF_U32(vcltq_f32(OB_F32(a), OB_F32(b))); }
OB_VFN v128_t wasm_f32x4_gt(v128_t a, v128_t b) { return OB_OF_U32(vcgtq_f32(OB_F32(a), OB_F32(b))); }
OB_VFN v128_t wasm_f32x4_le(v128_t a, v128_t b) { return OB_OF_U32(vcleq_f32(OB_F32(a), OB_F32(b))); }
// ai: int32 to float, to nearest even (SCVTF under the default rounding mode, which nothing here changes).
OB_VFN v128_t wasm_f32x4_convert_i32x4(v128_t a) { return OB_OF_F32(vcvtq_f32_s32(a)); }

// ai: i32x4. Float to int32 towards zero, saturating at both ends, NaN to 0: FCVTZS does all three.
OB_VFN v128_t wasm_i32x4_trunc_sat_f32x4(v128_t a) { return vcvtq_s32_f32(OB_F32(a)); }
OB_VFN v128_t wasm_i32x4_splat(int32_t a) { return vdupq_n_s32(a); }
// ai: Integer add, subtract and multiply wrap: done on unsigned lanes, the same bits.
OB_VFN v128_t wasm_i32x4_add(v128_t a, v128_t b) { return OB_OF_U32(vaddq_u32(OB_U32(a), OB_U32(b))); }
OB_VFN v128_t wasm_i32x4_mul(v128_t a, v128_t b) { return OB_OF_U32(vmulq_u32(OB_U32(a), OB_U32(b))); }
OB_VFN v128_t wasm_i32x4_min(v128_t a, v128_t b) { return vminq_s32(a, b); }
OB_VFN v128_t wasm_i32x4_max(v128_t a, v128_t b) { return vmaxq_s32(a, b); }
// ai: The lane number of an extract is a constant, so the extracts are macros over NEON's, which want one.
#define wasm_i32x4_extract_lane(a, i) vgetq_lane_s32((a), (i))
#define wasm_i64x2_extract_lane(a, i) vgetq_lane_s64(vreinterpretq_s64_s32(a), (i))
#define wasm_u8x16_extract_lane(a, i) vgetq_lane_u8(vreinterpretq_u8_s32(a), (i))
// ai: A bitmask is each lane's top bit, lane 0 at bit 0. NEON has no such instruction: the top bits are moved to
// ai: each lane's own bit position and the lanes summed.
OB_VFN uint32_t wasm_i32x4_bitmask(v128_t a) {
  static const int32_t at[4] = { 0, 1, 2, 3 };
  return vaddvq_u32(vshlq_u32(vshrq_n_u32(OB_U32(a), 31), vld1q_s32(at)));
}

// ai: i16x8 and u16x8.
OB_VFN v128_t wasm_i16x8_splat(int16_t a) { return OB_OF_S16(vdupq_n_s16(a)); }
OB_VFN v128_t wasm_i16x8_add(v128_t a, v128_t b) { return OB_OF_U16(vaddq_u16(OB_U16(a), OB_U16(b))); }
OB_VFN v128_t wasm_i16x8_sub(v128_t a, v128_t b) { return OB_OF_U16(vsubq_u16(OB_U16(a), OB_U16(b))); }
OB_VFN v128_t wasm_i16x8_mul(v128_t a, v128_t b) { return OB_OF_U16(vmulq_u16(OB_U16(a), OB_U16(b))); }
OB_VFN v128_t wasm_i16x8_min(v128_t a, v128_t b) { return OB_OF_S16(vminq_s16(OB_S16(a), OB_S16(b))); }
OB_VFN v128_t wasm_i16x8_max(v128_t a, v128_t b) { return OB_OF_S16(vmaxq_s16(OB_S16(a), OB_S16(b))); }
// ai: abs(-32768) stays -32768 (ABS, not the saturating SQABS).
OB_VFN v128_t wasm_i16x8_abs(v128_t a) { return OB_OF_S16(vabsq_s16(OB_S16(a))); }
OB_VFN v128_t wasm_i16x8_lt(v128_t a, v128_t b) { return OB_OF_U16(vcltq_s16(OB_S16(a), OB_S16(b))); }
OB_VFN v128_t wasm_i16x8_eq(v128_t a, v128_t b) { return OB_OF_U16(vceqq_s16(OB_S16(a), OB_S16(b))); }
// ai: Shifts take one count for every lane, taken modulo the lane's width. NEON shifts right by shifting left by a
// ai: negative count, and neither SSHL (arithmetic) nor USHL (logical) rounds.
OB_VFN v128_t wasm_i16x8_shr(v128_t a, uint32_t n) { return OB_OF_S16(vshlq_s16(OB_S16(a), vdupq_n_s16((int16_t)-(int)(n & 15)))); }
OB_VFN v128_t wasm_u16x8_shr(v128_t a, uint32_t n) { return OB_OF_U16(vshlq_u16(OB_U16(a), vdupq_n_s16((int16_t)-(int)(n & 15)))); }
// ai: Narrowing saturates: each signed 32-bit lane to a signed 16-bit one, a's lanes in the low half, b's above.
OB_VFN v128_t wasm_i16x8_narrow_i32x4(v128_t a, v128_t b) { return OB_OF_S16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))); }
// ai: The low or the high eight unsigned bytes, each widened to a 16-bit lane.
OB_VFN v128_t wasm_u16x8_extend_low_u8x16(v128_t a) { return OB_OF_U16(vmovl_u8(vget_low_u8(OB_U8(a)))); }
OB_VFN v128_t wasm_u16x8_extend_high_u8x16(v128_t a) { return OB_OF_U16(vmovl_u8(vget_high_u8(OB_U8(a)))); }
// ai: Neighbouring unsigned 16-bit lanes added into a 32-bit one (UADDLP).
OB_VFN v128_t wasm_u32x4_extadd_pairwise_u16x8(v128_t a) { return OB_OF_U32(vpaddlq_u16(OB_U16(a))); }

// ai: i8x16 and u8x16.
OB_VFN v128_t wasm_i8x16_splat(int8_t a) { return OB_OF_S8(vdupq_n_s8(a)); }
OB_VFN v128_t wasm_u8x16_splat(uint8_t a) { return OB_OF_U8(vdupq_n_u8(a)); }
OB_VFN v128_t wasm_u8x16_min(v128_t a, v128_t b) { return OB_OF_U8(vminq_u8(OB_U8(a), OB_U8(b))); }
OB_VFN v128_t wasm_u8x16_max(v128_t a, v128_t b) { return OB_OF_U8(vmaxq_u8(OB_U8(a), OB_U8(b))); }
OB_VFN v128_t wasm_u8x16_lt(v128_t a, v128_t b) { return OB_OF_U8(vcltq_u8(OB_U8(a), OB_U8(b))); }
OB_VFN v128_t wasm_i8x16_ne(v128_t a, v128_t b) { return OB_OF_U8(vmvnq_u8(vceqq_u8(OB_U8(a), OB_U8(b)))); }
// ai: a - b, 0 where b is the larger.
OB_VFN v128_t wasm_u8x16_sub_sat(v128_t a, v128_t b) { return OB_OF_U8(vqsubq_u8(OB_U8(a), OB_U8(b))); }
OB_VFN v128_t wasm_u8x16_shr(v128_t a, uint32_t n) { return OB_OF_U8(vshlq_u8(OB_U8(a), vdupq_n_s8((int8_t)-(int)(n & 7)))); }
// ai: Signed 16-bit lanes to bytes, saturating: to signed bytes, or (u8x16) to unsigned ones, a negative lane to 0.
OB_VFN v128_t wasm_i8x16_narrow_i16x8(v128_t a, v128_t b) { return OB_OF_S8(vcombine_s8(vqmovn_s16(OB_S16(a)), vqmovn_s16(OB_S16(b)))); }
OB_VFN v128_t wasm_u8x16_narrow_i16x8(v128_t a, v128_t b) { return OB_OF_U8(vcombine_u8(vqmovun_s16(OB_S16(a)), vqmovun_s16(OB_S16(b)))); }
// ai: Byte i of the result is byte s[i] of a, and 0 where s[i] is 16 or more (TBL answers 0 out of range).
OB_VFN v128_t wasm_i8x16_swizzle(v128_t a, v128_t s) { return OB_OF_U8(vqtbl1q_u8(OB_U8(a), OB_U8(s))); }
// ai: Sixteen constant indices into a's bytes (0 to 15) then b's (16 to 31).
#define wasm_i8x16_shuffle(a, b, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15) \
  vreinterpretq_s32_s8(__builtin_shufflevector(vreinterpretq_s8_s32(a), vreinterpretq_s8_s32(b), \
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15))
// ai: Sixteen top bits to sixteen mask bits: each byte keeps the bit of its place in its half, byte i is paired
// ai: with byte i + 8 into a 16-bit lane, and the eight lanes are summed.
OB_VFN uint32_t wasm_i8x16_bitmask(v128_t a) {
  const uint8x16_t bit = vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201ULL));
  const uint8x16_t m = vandq_u8(vcltzq_s8(OB_S8(a)), bit);
  return vaddvq_u16(vreinterpretq_u16_u8(vzip1q_u8(m, vextq_u8(m, m, 8))));
}

#undef OB_VFN
#undef OB_F32
#undef OB_S16
#undef OB_U16
#undef OB_S8
#undef OB_U8
#undef OB_U32
#undef OB_OF_F32
#undef OB_OF_S16
#undef OB_OF_U16
#undef OB_OF_S8
#undef OB_OF_U8
#undef OB_OF_U32
#endif
#endif
