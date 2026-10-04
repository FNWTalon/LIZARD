// ai: Does a vector build of the codec do what its scalar twins do? The vector paths are written against wasm's SIMD
// ai: intrinsics and run on arm64 through src/simd.h's NEON wrappers; this program is built twice from the same
// ai: source, as is (the vector paths) and with -DOB_SCALAR (the twins), and everything it prints except the lines
// ai: that begin "build" or "time" must be the same text from both. test/neon_check.sh builds the pair with the
// ai: NDK, runs them on a phone and compares.
// ai:   neon_check unit [full]             each simd.h wrapper against a plain C model of the wasm instruction, every
// ai:                                      special value against every other and random lanes; full adds the whole
// ai:                                      domain where that can be walked (every float through nearest, trunc_sat
// ai:                                      and abs, every int32 through convert, every 16-bit lane, every pair of
// ai:                                      bytes; a minute). A vector build only: on arm64 it holds the NEON wrappers
// ai:                                      to the model, and built by emcc with -msimd128 it holds the model to
// ai:                                      wasm's own instructions.
// ai:   neon_check synth                   digests of the vector sites no blind decode reaches (the transform at
// ai:                                      radix 4, RGBA to luma, the binarizer's ragged sizes, the interleaved RS
// ai:                                      syndromes, the encoder and its painter, the binary grid code), and every
// ai:                                      ring with every picture size painted, read blind and compared byte for
// ai:                                      byte. No input files.
// ai:   neon_check frames <list> [passes]  every frame of the list through focus_any_frame (one focus_any_t, nmax
// ai:                                      1536, gamma 1, mesh 2, as the receivers call it): a line a frame of
// ai:                                      digests, a stage each, and a total. passes > 0 then times that many
// ai:                                      passes over the list, one thread, ms a frame, with the core and clock
// ai:                                      each pass ended on.
// ai: A list is a line a frame, "w h held path": raw 8-bit luma, w x h of it; held is the version a receiver holds
// ai: for a frame whose word does not read (any.h), or -1 for the receiver's own rule, the last word read in the list.
// ai: A frame's digests (FNV-1a, 64 bits) are in the order the decode produces them, so the first that differs names
// ai: the stage:
// ai:   reg   the registration: found, finders, quad, orientation, track score, best mark, the ring, and the finder's
// ai:         own account of itself (acquire.c ob_finder_info, ob_raw_quad)
// ai:   word  the format word, the picture, whether the held word stood in
// ai:   spec  focus.c's test hook: the whole spectrum after the detrend, and every block's soft values
// ai:   stat  each block's LDPC iterations and its estimate
// ai:   blk   the ok flags and every block's bytes
// ai: No timing is in any of them. Exit 1 on a unit failure, a synth round trip that did not come back, or a build
// ai: that contracts a multiply and an add (-ffp-contract=off is missing).
// ai: From liblizard/ (C = the NDK's aarch64-linux-android29-clang; S = src/focus.c src/acquire.c src/layout.c src/fmt.c
// ai: src/rs.c src/ldpc.c src/fft.c src/any.c src/dec.c src/demap.c src/enc.c):
// ai:   $C -O3 -ffp-contract=off -DNDEBUG -Isrc -o build/neon/neon_check.neon test/neon_check.c $S -lm
// ai:   $C -O3 -ffp-contract=off -DNDEBUG -Isrc -DOB_SCALAR -o build/neon/neon_check.scalar test/neon_check.c $S -lm
// ai: It also builds with emcc (add -msimd128 or not, and -sNODERAWFS=1 -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1, run
// ai: by node) and with gcc on x86-64, scalar. The x86-64 build is good for synth; on a frames list it can die where a
// ai: line hypothesis hands sample() a NaN coordinate (internal.h casts it to int: INT_MIN there, 0 on arm64 and wasm).
#define _GNU_SOURCE   // ai: sched_getcpu, for the timing lines
#include "any.h"
#include "internal.h"
#include "fft.h"
#include "rs.h"
#include "simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
#include <sched.h>
#endif

extern uint32_t ob_debug_hash;                                            // ai: dec.c, read by focus.c
void ob_test_binarize(const uint8_t *img, int w, int h, uint8_t *bin);    // ai: acquire.c

#if defined(__wasm_simd128__)
#define BUILD_VECTOR "wasm simd128"
#elif defined(OB_SIMD)
#define BUILD_VECTOR "neon"
#else
#define BUILD_VECTOR "scalar"
#endif

typedef uint64_t dig_t;
#define DIG0 1469598103934665603ULL
static dig_t dig(dig_t h, const void *p, size_t n) { const uint8_t *b = p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; } return h; }
static dig_t dig_int(dig_t h, int v) { const int32_t x = v; return dig(h, &x, 4); }

// ai: One generator for every test input, integers only, so every build draws the same bytes.
static uint64_t rng_s;
static void rnd_seed(uint64_t s) { rng_s = s * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL; }
static uint32_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 16); }

static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
// ai: Where the thread is running and that core's clock, MHz (0 where the system does not say): nothing is pinned, so
// ai: a timing pass is the scheduler's choice of core and the governor's of clock, and the line says which it got.
static void where(int *cpu, int *mhz) {
  *cpu = -1; *mhz = 0;
#ifdef __linux__
  *cpu = sched_getcpu();
  char path[96];
  snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", *cpu);
  FILE *f = *cpu >= 0 ? fopen(path, "r") : NULL;
  long khz = 0;
  if (f) { if (fscanf(f, "%ld", &khz) != 1) khz = 0; fclose(f); }
  *mhz = (int)(khz / 1000);
#endif
}
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return 1e3 * (double)t.tv_sec + 1e-6 * (double)t.tv_nsec; }

// ai: (1 + 2^-23)^2 rounds to 1 + 2^-22, so the sum is 0 when the product is rounded before the add and 2^-46 when
// ai: the two are fused.
static int contracts(void) { volatile float a = 1.00000011920928955078125f, c = -1.0000002384185791015625f; const float r = a * a + c; return r != 0; }

// ai: ---------------------------------------------------------------- unit: simd.h against the wasm instructions

#ifdef OB_SIMD
typedef union { uint8_t u8[16]; int8_t i8[16]; uint16_t u16[8]; int16_t i16[8]; uint32_t u32[4]; int32_t i32[4]; float f32[4]; int64_t i64[2]; } lanes_t;
static v128_t vec(const lanes_t *a) { return wasm_v128_load(a); }
static lanes_t lanes(v128_t v) { lanes_t r; wasm_v128_store(&r, v); return r; }
static int u_checks, u_fails;

// ai: fl: the lanes are floats, where two NaNs agree whatever their bits (wasm leaves a NaN's payload open).
static void same(const char *name, lanes_t got, lanes_t want, int fl, const lanes_t *a, const lanes_t *b) {
  int ok = 1;
  u_checks++;
  if (fl) { for (int i = 0; i < 4; i++) ok &= got.u32[i] == want.u32[i] || (got.f32[i] != got.f32[i] && want.f32[i] != want.f32[i]); }
  else ok = !memcmp(&got, &want, 16);
  if (ok) return;
  if (u_fails++ < 40) {
    printf("  FAIL %s: got %08x %08x %08x %08x want %08x %08x %08x %08x", name, got.u32[0], got.u32[1], got.u32[2], got.u32[3], want.u32[0], want.u32[1], want.u32[2], want.u32[3]);
    if (a) printf(" a %08x %08x %08x %08x", a->u32[0], a->u32[1], a->u32[2], a->u32[3]);
    if (b) printf(" b %08x %08x %08x %08x", b->u32[0], b->u32[1], b->u32[2], b->u32[3]);
    printf("\n");
  }
}
static void same_u32(const char *name, uint32_t got, uint32_t want) { u_checks++; if (got != want && u_fails++ < 40) printf("  FAIL %s: got %08x want %08x\n", name, got, want); }

// ai: Floats a twin can tell apart: zeros of both signs, halves (ties), the int32 range's ends, 2^23 (where a float
// ai: stops having a fraction), infinities, NaNs quiet and signalling, the smallest and largest numbers.
static const uint32_t F_SPECIAL[] = {
  0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000, 0x40200000, 0xc0200000,
  0x40600000, 0xc0600000, 0x3effffff, 0x3f000001, 0xbeffffff, 0xbf000001, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
  0x7fa00000, 0xffa00000, 0x7fc12345, 0x00000001, 0x80000001, 0x007fffff, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
  0x4f000000, 0xcf000000, 0x4effffff, 0xceffffff, 0xcf000001, 0x4f000001, 0x4b000000, 0xcb000000, 0x4affffff, 0xcaffffff,
  0x4a7ffffe, 0x437f0000, 0x437f8000, 0x43800000, 0x5f000000, 0xdf000000, 0x3f7fffff, 0x3f800001, 0x41200000, 0xc1200000 };
static const uint32_t I_SPECIAL[] = { 0, 1, 0xffffffff, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0x80008000, 0x7fff7fff,
  0x80808080, 0x7f7f7f7f, 0x00ff00ff, 0xff00ff00, 0x01010101, 0xfffefffe, 0x00010001, 0x0f0f0f0f, 0xf0f0f0f0, 0x10101010, 0x1f1f1f1f, 0x8000ffff, 0xffff8000 };
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t u_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static lanes_t draw_f(void) {
  lanes_t r;
  for (int i = 0; i < 4; i++) {
    const uint32_t k = rnd() % 4;
    if (k == 0) r.u32[i] = F_SPECIAL[rnd() % (sizeof F_SPECIAL / 4)];
    else if (k == 1) r.u32[i] = rnd();
    else if (k == 2) r.f32[i] = (float)((int)(rnd() % 8001) - 4000) * 0.25f;          // ai: quarters to 1000: ties and near ties
    else r.f32[i] = (float)((int)(rnd() & 0xffffff) - 0x800000) * (1.0f / 4096.0f);   // ai: twelve fraction bits to 2048
  }
  return r;
}
static lanes_t draw_i(void) {
  lanes_t r;
  for (int i = 0; i < 4; i++) r.u32[i] = rnd() % 3 == 0 ? I_SPECIAL[rnd() % (sizeof I_SPECIAL / 4)] : rnd();
  return r;
}
// ai: The models. Each is the wasm SIMD instruction's definition for one lane, in C a scalar build would compile, and
// ai: kept out of line so that it stays scalar code beside the vector instruction it is held against.
#define MODEL static __attribute__((__noinline__))
MODEL float m_nearest(float x) {
  const float a = x < 0 ? -x : x;
  if (x != x || !(a < 8388608.0f)) return x;                  // ai: NaN, or already whole (2^23 and up, the infinities)
  const float r = (a + 8388608.0f) - 8388608.0f;              // ai: to the nearest whole number, halves to even
  return f_of(u_of(r) | (u_of(x) & 0x80000000u));             // ai: the operand's sign, so -0.25 gives -0
}
MODEL int32_t m_trunc_sat(float x) { return x != x ? 0 : x >= 2147483648.0f ? INT32_MAX : x <= -2147483648.0f ? INT32_MIN : (int32_t)x; }
MODEL int16_t m_sar16(int16_t a, unsigned n) { n &= 15; return (int16_t)(a < 0 ? ~(~(int)a >> n) : (int)a >> n); }

// ai: The float operations on one pair of vectors.
static void unit_f(const lanes_t a, const lanes_t b) {
    const v128_t va = vec(&a), vb = vec(&b);
    lanes_t w;
    #define F2(name, expr) do { for (int i = 0; i < 4; i++) { const float x = a.f32[i], y = b.f32[i]; (void)y; w.f32[i] = (expr); } same(#name, lanes(name(va, vb)), w, 1, &a, &b); } while (0)
    F2(wasm_f32x4_add, x + y); F2(wasm_f32x4_sub, x - y); F2(wasm_f32x4_mul, x * y); F2(wasm_f32x4_div, x / y);
    #undef F2
    // ai: pmin and pmax hand back one operand's bits, a NaN's included, so they are held to the bit.
    for (int i = 0; i < 4; i++) w.u32[i] = b.f32[i] < a.f32[i] ? b.u32[i] : a.u32[i];
    same("wasm_f32x4_pmin", lanes(wasm_f32x4_pmin(va, vb)), w, 0, &a, &b);
    for (int i = 0; i < 4; i++) w.u32[i] = a.f32[i] < b.f32[i] ? b.u32[i] : a.u32[i];
    same("wasm_f32x4_pmax", lanes(wasm_f32x4_pmax(va, vb)), w, 0, &a, &b);
    for (int i = 0; i < 4; i++) w.u32[i] = a.f32[i] < b.f32[i] ? 0xffffffffu : 0;
    same("wasm_f32x4_lt", lanes(wasm_f32x4_lt(va, vb)), w, 0, &a, &b);
    for (int i = 0; i < 4; i++) w.u32[i] = a.f32[i] > b.f32[i] ? 0xffffffffu : 0;
    same("wasm_f32x4_gt", lanes(wasm_f32x4_gt(va, vb)), w, 0, &a, &b);
    for (int i = 0; i < 4; i++) w.u32[i] = a.f32[i] <= b.f32[i] ? 0xffffffffu : 0;
    same("wasm_f32x4_le", lanes(wasm_f32x4_le(va, vb)), w, 0, &a, &b);
    for (int i = 0; i < 4; i++) w.u32[i] = a.u32[i] & 0x7fffffffu;
    same("wasm_f32x4_abs", lanes(wasm_f32x4_abs(va)), w, 0, &a, 0);
    for (int i = 0; i < 4; i++) w.f32[i] = m_nearest(a.f32[i]);
    same("wasm_f32x4_nearest", lanes(wasm_f32x4_nearest(va)), w, 1, &a, 0);
    for (int i = 0; i < 4; i++) w.i32[i] = m_trunc_sat(a.f32[i]);
    same("wasm_i32x4_trunc_sat_f32x4", lanes(wasm_i32x4_trunc_sat_f32x4(va)), w, 0, &a, 0);
    same("wasm_f32x4_make", lanes(wasm_f32x4_make(a.f32[0], a.f32[1], a.f32[2], a.f32[3])), a, 1, &a, 0);
    for (int i = 0; i < 4; i++) w.f32[i] = b.f32[2];
    same("wasm_f32x4_splat", lanes(wasm_f32x4_splat(b.f32[2])), w, 1, &b, 0);
}
// ai: The integer, bit and memory operations on one triple. it: the draw's number, which picks the shift count's kind.
static void unit_i(const lanes_t a, const lanes_t b, const lanes_t c, int it) {
    const v128_t va = vec(&a), vb = vec(&b), vc = vec(&c);
    lanes_t w;
    // ai: Memory: sixteen bytes to and from any address.
    { uint8_t buf[40]; lanes_t r; memset(buf, 0xA5, sizeof buf); const int o = 1 + (int)(rnd() % 15);
      memcpy(buf + o, &a, 16); same("wasm_v128_load", lanes(wasm_v128_load(buf + o)), a, 0, &a, 0);
      memset(buf, 0x5A, sizeof buf); wasm_v128_store(buf + o, va); memcpy(&r, buf + o, 16); same("wasm_v128_store", r, a, 0, &a, 0);
      same_u32("wasm_v128_store leaves the bytes either side", (uint32_t)(buf[o - 1] == 0x5A && buf[o + 16] == 0x5A), 1);
      for (int i = 0; i < 8; i++) w.i16[i] = (int16_t)a.i8[i + 3];
      memcpy(buf + o, &a, 16); same("wasm_i16x8_load8x8", lanes(wasm_i16x8_load8x8(buf + o + 3)), w, 0, &a, 0); }
    #define OP(name, lane, n, expr) do { for (int i = 0; i < n; i++) w.lane[i] = (expr); same(#name, lanes(name(va, vb)), w, 0, &a, &b); } while (0)
    OP(wasm_v128_and, u32, 4, a.u32[i] & b.u32[i]); OP(wasm_v128_or, u32, 4, a.u32[i] | b.u32[i]); OP(wasm_v128_xor, u32, 4, a.u32[i] ^ b.u32[i]);
    for (int i = 0; i < 4; i++) w.u32[i] = (a.u32[i] & c.u32[i]) | (b.u32[i] & ~c.u32[i]);
    same("wasm_v128_bitselect", lanes(wasm_v128_bitselect(va, vb, vc)), w, 0, &a, &b);
    OP(wasm_i32x4_add, u32, 4, a.u32[i] + b.u32[i]); OP(wasm_i32x4_mul, u32, 4, a.u32[i] * b.u32[i]);
    OP(wasm_i32x4_min, i32, 4, a.i32[i] < b.i32[i] ? a.i32[i] : b.i32[i]); OP(wasm_i32x4_max, i32, 4, a.i32[i] > b.i32[i] ? a.i32[i] : b.i32[i]);
    for (int i = 0; i < 4; i++) w.f32[i] = (float)a.i32[i];
    same("wasm_f32x4_convert_i32x4", lanes(wasm_f32x4_convert_i32x4(va)), w, 0, &a, 0);
    for (int i = 0; i < 4; i++) w.i32[i] = b.i32[1];
    same("wasm_i32x4_splat", lanes(wasm_i32x4_splat(b.i32[1])), w, 0, &b, 0);
    same_u32("wasm_i32x4_extract_lane 0", (uint32_t)wasm_i32x4_extract_lane(va, 0), a.u32[0]); same_u32("wasm_i32x4_extract_lane 1", (uint32_t)wasm_i32x4_extract_lane(va, 1), a.u32[1]);
    same_u32("wasm_i32x4_extract_lane 2", (uint32_t)wasm_i32x4_extract_lane(va, 2), a.u32[2]); same_u32("wasm_i32x4_extract_lane 3", (uint32_t)wasm_i32x4_extract_lane(va, 3), a.u32[3]);
    { const int64_t e0 = wasm_i64x2_extract_lane(va, 0), e1 = wasm_i64x2_extract_lane(va, 1); u_checks++; if (e0 != a.i64[0] || e1 != a.i64[1]) { if (u_fails++ < 40) printf("  FAIL wasm_i64x2_extract_lane\n"); } }
    { uint32_t m = 0; for (int i = 0; i < 4; i++) m |= (a.u32[i] >> 31) << i; same_u32("wasm_i32x4_bitmask", wasm_i32x4_bitmask(va), m); }
    OP(wasm_i16x8_add, u16, 8, (uint16_t)(a.u16[i] + b.u16[i])); OP(wasm_i16x8_sub, u16, 8, (uint16_t)(a.u16[i] - b.u16[i])); OP(wasm_i16x8_mul, u16, 8, (uint16_t)((uint32_t)a.u16[i] * b.u16[i]));
    OP(wasm_i16x8_min, i16, 8, a.i16[i] < b.i16[i] ? a.i16[i] : b.i16[i]); OP(wasm_i16x8_max, i16, 8, a.i16[i] > b.i16[i] ? a.i16[i] : b.i16[i]);
    OP(wasm_i16x8_lt, u16, 8, a.i16[i] < b.i16[i] ? 0xffff : 0); OP(wasm_i16x8_eq, u16, 8, a.i16[i] == b.i16[i] ? 0xffff : 0);
    for (int i = 0; i < 8; i++) w.u16[i] = a.i16[i] < 0 ? (uint16_t)(0u - a.u16[i]) : a.u16[i];   // ai: -32768 stays -32768
    same("wasm_i16x8_abs", lanes(wasm_i16x8_abs(va)), w, 0, &a, 0);
    for (int i = 0; i < 8; i++) w.i16[i] = b.i16[5];
    same("wasm_i16x8_splat", lanes(wasm_i16x8_splat(b.i16[5])), w, 0, &b, 0);
    // ai: Shifts: a count the compiler cannot see, any value (taken modulo the lane's width), and the constants the source uses.
    { volatile uint32_t n = it % 3 == 0 ? rnd() : rnd() % 40; const uint32_t k = n;
      for (int i = 0; i < 8; i++) w.i16[i] = m_sar16(a.i16[i], k);
      same("wasm_i16x8_shr", lanes(wasm_i16x8_shr(va, k)), w, 0, &a, 0);
      for (int i = 0; i < 8; i++) w.u16[i] = (uint16_t)(a.u16[i] >> (k & 15));
      same("wasm_u16x8_shr", lanes(wasm_u16x8_shr(va, k)), w, 0, &a, 0);
      for (int i = 0; i < 16; i++) w.u8[i] = (uint8_t)(a.u8[i] >> (k & 7));
      same("wasm_u8x16_shr", lanes(wasm_u8x16_shr(va, k)), w, 0, &a, 0);
      for (int i = 0; i < 8; i++) w.i16[i] = m_sar16(a.i16[i], 4);
      same("wasm_i16x8_shr 4", lanes(wasm_i16x8_shr(va, 4)), w, 0, &a, 0);
      for (int i = 0; i < 8; i++) w.i16[i] = m_sar16(a.i16[i], 15);
      same("wasm_i16x8_shr 15", lanes(wasm_i16x8_shr(va, 15)), w, 0, &a, 0);
      for (int i = 0; i < 8; i++) w.u16[i] = (uint16_t)(a.u16[i] >> 8);
      same("wasm_u16x8_shr 8", lanes(wasm_u16x8_shr(va, 8)), w, 0, &a, 0);
      for (int i = 0; i < 16; i++) w.u8[i] = (uint8_t)(a.u8[i] >> 4);
      same("wasm_u8x16_shr 4", lanes(wasm_u8x16_shr(va, 4)), w, 0, &a, 0); }
    for (int i = 0; i < 4; i++) { w.i16[i] = (int16_t)clampi(a.i32[i], -32768, 32767); w.i16[4 + i] = (int16_t)clampi(b.i32[i], -32768, 32767); }
    same("wasm_i16x8_narrow_i32x4", lanes(wasm_i16x8_narrow_i32x4(va, vb)), w, 0, &a, &b);
    for (int i = 0; i < 8; i++) { w.i8[i] = (int8_t)clampi(a.i16[i], -128, 127); w.i8[8 + i] = (int8_t)clampi(b.i16[i], -128, 127); }
    same("wasm_i8x16_narrow_i16x8", lanes(wasm_i8x16_narrow_i16x8(va, vb)), w, 0, &a, &b);
    for (int i = 0; i < 8; i++) { w.u8[i] = (uint8_t)clampi(a.i16[i], 0, 255); w.u8[8 + i] = (uint8_t)clampi(b.i16[i], 0, 255); }
    same("wasm_u8x16_narrow_i16x8", lanes(wasm_u8x16_narrow_i16x8(va, vb)), w, 0, &a, &b);
    for (int i = 0; i < 8; i++) w.u16[i] = a.u8[i];
    same("wasm_u16x8_extend_low_u8x16", lanes(wasm_u16x8_extend_low_u8x16(va)), w, 0, &a, 0);
    for (int i = 0; i < 8; i++) w.u16[i] = a.u8[8 + i];
    same("wasm_u16x8_extend_high_u8x16", lanes(wasm_u16x8_extend_high_u8x16(va)), w, 0, &a, 0);
    for (int i = 0; i < 4; i++) w.u32[i] = (uint32_t)a.u16[2 * i] + a.u16[2 * i + 1];
    same("wasm_u32x4_extadd_pairwise_u16x8", lanes(wasm_u32x4_extadd_pairwise_u16x8(va)), w, 0, &a, 0);
    OP(wasm_u8x16_min, u8, 16, a.u8[i] < b.u8[i] ? a.u8[i] : b.u8[i]); OP(wasm_u8x16_max, u8, 16, a.u8[i] > b.u8[i] ? a.u8[i] : b.u8[i]);
    OP(wasm_u8x16_lt, u8, 16, a.u8[i] < b.u8[i] ? 0xff : 0); OP(wasm_i8x16_ne, u8, 16, a.u8[i] != b.u8[i] ? 0xff : 0);
    OP(wasm_u8x16_sub_sat, u8, 16, (uint8_t)(a.u8[i] > b.u8[i] ? a.u8[i] - b.u8[i] : 0));
    // ai: ne on lanes that mostly agree, as the run scans meet them
    { lanes_t d = a; d.u8[rnd() % 16] ^= (uint8_t)(1 + rnd() % 255); const v128_t vd = vec(&d);
      for (int i = 0; i < 16; i++) w.u8[i] = a.u8[i] != d.u8[i] ? 0xff : 0;
      same("wasm_i8x16_ne, one lane off", lanes(wasm_i8x16_ne(va, vd)), w, 0, &a, &d);
      uint32_t m = 0; for (int i = 0; i < 16; i++) m |= (uint32_t)(a.u8[i] != d.u8[i]) << i;
      same_u32("wasm_i8x16_bitmask of ne", wasm_i8x16_bitmask(wasm_i8x16_ne(va, vd)), m); }
    #undef OP
    for (int i = 0; i < 16; i++) w.i8[i] = b.i8[9];
    same("wasm_i8x16_splat", lanes(wasm_i8x16_splat(b.i8[9])), w, 0, &b, 0);
    same("wasm_u8x16_splat", lanes(wasm_u8x16_splat(b.u8[9])), w, 0, &b, 0);
    { uint32_t m = 0; for (int i = 0; i < 16; i++) m |= (uint32_t)(a.u8[i] >> 7) << i; same_u32("wasm_i8x16_bitmask", wasm_i8x16_bitmask(va), m); }
    // ai: An unsigned extract of a byte of 128 or more stays positive.
    same_u32("wasm_u8x16_extract_lane 0", wasm_u8x16_extract_lane(va, 0), a.u8[0]); same_u32("wasm_u8x16_extract_lane 8", wasm_u8x16_extract_lane(va, 8), a.u8[8]);
    same_u32("wasm_u8x16_extract_lane 15", wasm_u8x16_extract_lane(va, 15), a.u8[15]);
    // ai: swizzle: indices of every kind, 16 and up (and the ones with the top bit set) giving 0
    { lanes_t s; for (int i = 0; i < 16; i++) { const uint32_t k = rnd() % 4; s.u8[i] = (uint8_t)(k == 0 ? rnd() : k == 1 ? 16 + rnd() % 3 : rnd() % 16); }
      for (int i = 0; i < 16; i++) w.u8[i] = s.u8[i] < 16 ? a.u8[s.u8[i]] : 0;
      same("wasm_i8x16_swizzle", lanes(wasm_i8x16_swizzle(va, vec(&s))), w, 0, &a, &s); }
    // ai: shuffle: the index lists the source uses, and three that draw on both operands every way
    #define SH(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15) do { const int ix[16] = { c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 }; \
      for (int i = 0; i < 16; i++) w.u8[i] = ix[i] < 16 ? a.u8[ix[i]] : b.u8[ix[i] - 16]; \
      same("wasm_i8x16_shuffle " #c0 " " #c1 " " #c2 " " #c3 " ...", lanes(wasm_i8x16_shuffle(va, vb, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15)), w, 0, &a, &b); } while (0)
    SH(0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29); SH(2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0);
    SH(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23); SH(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
    SH(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14); SH(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13); SH(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
    SH(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16); SH(15, 16, 14, 17, 13, 18, 12, 19, 11, 20, 10, 21, 9, 22, 8, 23); SH(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    #undef SH
}
// ai: The whole domain where it is small enough to walk: every float through the three one-operand float operations
// ai: and every int32 through the conversion (2^32 each, a minute on a phone), every 16-bit lane through abs, the
// ai: narrowings and every shift count, every pair of bytes through the byte operations, every swizzle index in every
// ai: lane, every bitmask.
static void unit_full(void) {
  uint64_t bad[4] = { 0, 0, 0, 0 };
  for (uint64_t u = 0; u < (1ULL << 32); u += 4) {
    lanes_t a;
    for (int l = 0; l < 4; l++) a.u32[l] = (uint32_t)u + (uint32_t)l;
    const v128_t va = vec(&a);
    const lanes_t n = lanes(wasm_f32x4_nearest(va)), t = lanes(wasm_i32x4_trunc_sat_f32x4(va)), ab = lanes(wasm_f32x4_abs(va)), cv = lanes(wasm_f32x4_convert_i32x4(va));
    for (int l = 0; l < 4; l++) {
      const float x = a.f32[l], mn = m_nearest(x);
      bad[0] += !(n.u32[l] == u_of(mn) || (mn != mn && n.f32[l] != n.f32[l]));
      bad[1] += t.i32[l] != m_trunc_sat(x);
      bad[2] += ab.u32[l] != (a.u32[l] & 0x7fffffffu);
      bad[3] += cv.u32[l] != u_of((float)a.i32[l]);
    }
  }
  static const char *const NAME[4] = { "wasm_f32x4_nearest", "wasm_i32x4_trunc_sat_f32x4", "wasm_f32x4_abs", "wasm_f32x4_convert_i32x4" };
  for (int k = 0; k < 4; k++) { u_checks++; if (bad[k]) { u_fails++; printf("  FAIL %s over every input: %llu lanes differ\n", NAME[k], (unsigned long long)bad[k]); } }
  for (int v = -32768; v <= 32767; v++) {
    lanes_t a, b, w;
    for (int l = 0; l < 8; l++) { a.i16[l] = (int16_t)v; b.i16[l] = (int16_t)(v ^ (l * 0x1111)); }
    const v128_t va = vec(&a), vb = vec(&b);
    for (int l = 0; l < 8; l++) w.u16[l] = a.i16[l] < 0 ? (uint16_t)(0u - a.u16[l]) : a.u16[l];
    same("wasm_i16x8_abs, every lane value", lanes(wasm_i16x8_abs(va)), w, 0, &a, 0);
    for (int l = 0; l < 8; l++) { w.i8[l] = (int8_t)clampi(a.i16[l], -128, 127); w.i8[8 + l] = (int8_t)clampi(b.i16[l], -128, 127); }
    same("wasm_i8x16_narrow_i16x8, every lane value", lanes(wasm_i8x16_narrow_i16x8(va, vb)), w, 0, &a, &b);
    for (int l = 0; l < 8; l++) { w.u8[l] = (uint8_t)clampi(a.i16[l], 0, 255); w.u8[8 + l] = (uint8_t)clampi(b.i16[l], 0, 255); }
    same("wasm_u8x16_narrow_i16x8, every lane value", lanes(wasm_u8x16_narrow_i16x8(va, vb)), w, 0, &a, &b);
    // ai: 32-bit lanes either side of the 16-bit range, and far outside it
    lanes_t c, d;
    for (int l = 0; l < 4; l++) { c.i32[l] = 3 * v + l - 1; d.i32[l] = v * 65537 + l; }
    for (int l = 0; l < 4; l++) { w.i16[l] = (int16_t)clampi(c.i32[l], -32768, 32767); w.i16[4 + l] = (int16_t)clampi(d.i32[l], -32768, 32767); }
    same("wasm_i16x8_narrow_i32x4, across the range", lanes(wasm_i16x8_narrow_i32x4(vec(&c), vec(&d))), w, 0, &c, &d);
    for (volatile uint32_t n = 0; n < 32; n++) {
      const uint32_t k = n;
      for (int l = 0; l < 8; l++) w.i16[l] = m_sar16(a.i16[l], k);
      same("wasm_i16x8_shr, every lane value and count", lanes(wasm_i16x8_shr(va, k)), w, 0, &a, 0);
      for (int l = 0; l < 8; l++) w.u16[l] = (uint16_t)(a.u16[l] >> (k & 15));
      same("wasm_u16x8_shr, every lane value and count", lanes(wasm_u16x8_shr(va, k)), w, 0, &a, 0);
    }
    // ai: the value as a mask: bit i set makes byte i negative
    for (int l = 0; l < 16; l++) a.u8[l] = (uint8_t)((((unsigned)v >> l) & 1) << 7 | ((unsigned)(v * 31 + l * 7) & 0x7f));
    same_u32("wasm_i8x16_bitmask, every mask", wasm_i8x16_bitmask(vec(&a)), (uint32_t)v & 0xffff);
  }
  for (int x = 0; x < 256; x++) {
    lanes_t a, b, w;
    for (int y = 0; y < 256; y += 16) {
      for (int l = 0; l < 16; l++) { a.u8[l] = (uint8_t)x; b.u8[l] = (uint8_t)(y + l); }
      const v128_t va = vec(&a), vb = vec(&b);
      #define OP(name, expr) do { for (int l = 0; l < 16; l++) w.u8[l] = (uint8_t)(expr); same(#name ", every pair of bytes", lanes(name(va, vb)), w, 0, &a, &b); } while (0)
      OP(wasm_u8x16_min, a.u8[l] < b.u8[l] ? a.u8[l] : b.u8[l]); OP(wasm_u8x16_max, a.u8[l] > b.u8[l] ? a.u8[l] : b.u8[l]);
      OP(wasm_u8x16_lt, a.u8[l] < b.u8[l] ? 0xff : 0); OP(wasm_i8x16_ne, a.u8[l] != b.u8[l] ? 0xff : 0); OP(wasm_u8x16_sub_sat, a.u8[l] > b.u8[l] ? a.u8[l] - b.u8[l] : 0);
      #undef OP
      // ai: b as indices into a table of sixteen distinct bytes
      for (int l = 0; l < 16; l++) a.u8[l] = (uint8_t)(x * 16 + l + 1);
      for (int l = 0; l < 16; l++) w.u8[l] = b.u8[l] < 16 ? a.u8[b.u8[l]] : 0;
      same("wasm_i8x16_swizzle, every index", lanes(wasm_i8x16_swizzle(vec(&a), vb)), w, 0, &a, &b);
    }
    for (int l = 0; l < 16; l++) a.u8[l] = (uint8_t)(x + 17 * l);
    for (volatile uint32_t n = 0; n < 16; n++) {
      const uint32_t k = n;
      for (int l = 0; l < 16; l++) w.u8[l] = (uint8_t)(a.u8[l] >> (k & 7));
      same("wasm_u8x16_shr, every byte and count", lanes(wasm_u8x16_shr(vec(&a), k)), w, 0, &a, 0);
    }
    for (int l = 0; l < 8; l++) w.i16[l] = (int16_t)a.i8[l];
    same("wasm_i16x8_load8x8, every byte", lanes(wasm_i16x8_load8x8(&a)), w, 0, &a, 0);
    for (int l = 0; l < 8; l++) w.u16[l] = a.u8[l];
    same("wasm_u16x8_extend_low_u8x16, every byte", lanes(wasm_u16x8_extend_low_u8x16(vec(&a))), w, 0, &a, 0);
    for (int l = 0; l < 8; l++) w.u16[l] = a.u8[8 + l];
    same("wasm_u16x8_extend_high_u8x16, every byte", lanes(wasm_u16x8_extend_high_u8x16(vec(&a))), w, 0, &a, 0);
  }
  for (uint32_t m = 0; m < 16; m++) {
    lanes_t a;
    for (int l = 0; l < 4; l++) a.u32[l] = ((m >> l) & 1) << 31 | (0x12345678u * (m + 1) + (uint32_t)l) >> 1;
    same_u32("wasm_i32x4_bitmask, every mask", wasm_i32x4_bitmask(vec(&a)), m);
  }
}
// ai: Every special value against every other in all four lanes, then random draws with specials among them; full: then
// ai: the whole domain of the operations that have a small one.
static int unit(int full) {
  enum { N = 20000, NF = sizeof F_SPECIAL / 4, NI = sizeof I_SPECIAL / 4 };
  rnd_seed(1);
  for (int i = 0; i < NF; i++) for (int j = 0; j < NF; j++) { lanes_t a, b; for (int l = 0; l < 4; l++) { a.u32[l] = F_SPECIAL[i]; b.u32[l] = F_SPECIAL[j]; } unit_f(a, b); }
  for (int i = 0; i < NI; i++) for (int j = 0; j < NI; j++) { lanes_t a, b, c; for (int l = 0; l < 4; l++) { a.u32[l] = I_SPECIAL[i]; b.u32[l] = I_SPECIAL[j]; c.u32[l] = I_SPECIAL[(i + 3 * j + l) % NI]; } unit_i(a, b, c, i * NI + j); }
  for (int it = 0; it < N; it++) { const lanes_t a = draw_f(), b = draw_f(); unit_f(a, b); }
  for (int it = 0; it < N; it++) { const lanes_t a = draw_i(), b = draw_i(), c = draw_i(); unit_i(a, b, c, it); }
  if (full) unit_full();
  printf("unit%s: %d checks of simd.h against the wasm instructions' model, %d failures\n", full ? " full" : "", u_checks, u_fails);
  return u_fails != 0;
}
#else
static int unit(int full) { (void)full; printf("unit: a scalar build has no vector paths to check\n"); return 0; }
#endif

// ai: ---------------------------------------------------------------- one frame, read blind

typedef struct { focus_any_t any; int top, B; uint8_t *blocks, *ok; } rx_t;
// ai: via: what gave the finder its quad (acquire.c ob_finder_info[4]): 0 the corner marks, 1 the line fit, 2 the crop
// ai: round one mark, -1 nothing.
typedef struct { dig_t reg, word, stat, blk; uint32_t spec; int found, via, ring, n, held, has_word, version, fps, got, total; float ms_detect, ms_sample, ms_decode; } out_t;

static int rx_init(rx_t *rx) {
  memset(rx, 0, sizeof *rx);
  rx->any.which = -1;
  rx->B = focus_any_init(&rx->any, 1536);
  if (rx->B < 0) return -1;
  rx->top = focus_any_top(&rx->any);
  rx->blocks = malloc((size_t)rx->top * rx->B); rx->ok = malloc((size_t)rx->top);
  return rx->blocks && rx->ok ? 0 : -1;
}
static void rx_free(rx_t *rx) { focus_any_free(&rx->any); free(rx->blocks); free(rx->ok); }

// ai: arm: take focus.c's hash of the spectrum and the soft values, which costs a pass over both (off when timing).
static void decode_one(rx_t *rx, const uint8_t *px, int w, int h, int held, int arm, out_t *o) {
  focus_any_t *a = &rx->any;
  ob_result_t res;
  memset(o, 0, sizeof *o);
  memset(rx->blocks, 0, (size_t)rx->top * rx->B); memset(rx->ok, 0, (size_t)rx->top);
  ob_debug_hash = arm ? 1 : 0;
  for (int k = 0; k < 8; k++) ob_finder_info[k] = -2;
  o->got = focus_any_frame(a, px, w, h, 1.0f, 2, rx->blocks, rx->ok, held, &res);
  o->spec = arm && ob_debug_hash != 1 ? ob_debug_hash : 0;   // ai: still 1: the frame never reached the transform
  ob_debug_hash = 0;
  const ob_fmt_t *fm = focus_any_word(a);
  o->found = res.found; o->via = (int)ob_finder_info[4]; o->ring = a->which; o->n = a->n; o->held = a->held; o->has_word = fm != 0; o->version = fm ? fm->version : 0; o->fps = fm ? fm->fps : 0;
  o->ms_detect = res.ms_detect; o->ms_sample = res.ms_sample; o->ms_decode = res.ms_decode;
  if (!arm) return;
  dig_t d = DIG0;
  d = dig_int(d, res.found); d = dig_int(d, res.finders); d = dig(d, res.quad, sizeof res.quad); d = dig_int(d, res.orient);
  d = dig(d, &res.mark_score, sizeof res.mark_score); d = dig(d, res.mark, sizeof res.mark); d = dig_int(d, a->which);
  d = dig(d, ob_finder_info, 8 * sizeof(float)); d = dig(d, ob_raw_quad, 8 * sizeof(float));
  o->reg = d;
  d = DIG0; d = dig_int(d, o->has_word); d = dig_int(d, o->version); d = dig_int(d, o->fps); d = dig_int(d, a->n); d = dig_int(d, a->held);
  o->word = d;
  d = DIG0;
  if (a->last) {
    const int8_t *its; const float *est;
    focus_block_stats(a->last, &its, &est);
    o->total = a->last->blocks;
    d = dig(d, its, (size_t)o->total); d = dig(d, est, (size_t)o->total * sizeof(float));
  }
  o->stat = d;
  d = DIG0; d = dig_int(d, o->got); d = dig_int(d, res.tiles_ok); d = dig(d, res.ok, sizeof res.ok); d = dig(d, res.iters, sizeof res.iters);
  d = dig(d, &res.ber, sizeof res.ber); d = dig(d, &res.gmi, sizeof res.gmi); d = dig(d, rx->ok, (size_t)rx->top); d = dig(d, rx->blocks, (size_t)rx->top * rx->B);
  o->blk = d;
}
static dig_t out_line(const char *kind, int idx, const char *name, const out_t *o, dig_t total) {
  printf("%s %04d %-26s reg %016llx word %016llx spec %08x stat %016llx blk %016llx  found %d via %s ring %d n %d word %d/%d held %d blocks %d/%d\n", kind, idx, name,
    (unsigned long long)o->reg, (unsigned long long)o->word, o->spec, (unsigned long long)o->stat, (unsigned long long)o->blk,
    o->found, o->via == 0 ? "marks" : o->via == 1 ? "lines" : o->via == 2 ? "crop" : "none", o->ring, o->n, o->version, o->fps, o->held, o->got, o->total);
  total = dig(total, &o->reg, sizeof o->reg); total = dig(total, &o->word, sizeof o->word); total = dig(total, &o->spec, sizeof o->spec);
  total = dig(total, &o->stat, sizeof o->stat); return dig(total, &o->blk, sizeof o->blk);
}

// ai: ---------------------------------------------------------------- frames

typedef struct { int w, h, held; char path[512], name[64]; uint8_t *px; } frame_t;

static int frames(const char *list, int passes) {
  FILE *fl = fopen(list, "r");
  if (!fl) { printf("cannot open %s\n", list); return 1; }
  frame_t *fr = NULL;
  int nf = 0, cap = 0;
  char line[700];
  while (fgets(line, sizeof line, fl)) {
    frame_t f;
    memset(&f, 0, sizeof f);
    if (sscanf(line, "%d %d %d %511s", &f.w, &f.h, &f.held, f.path) != 4) continue;
    const char *base = strrchr(f.path, '/'); base = base ? base + 1 : f.path;
    snprintf(f.name, sizeof f.name, "%.63s", base);
    char *dot = strrchr(f.name, '.'); if (dot) *dot = 0;
    FILE *fp = fopen(f.path, "rb");
    if (!fp) { printf("cannot open %s\n", f.path); return 1; }
    f.px = malloc((size_t)f.w * f.h);
    if (!f.px || fread(f.px, 1, (size_t)f.w * f.h, fp) != (size_t)f.w * f.h || fgetc(fp) != EOF) { printf("%s is not %d x %d bytes\n", f.path, f.w, f.h); return 1; }
    fclose(fp);
    if (nf == cap) { cap = cap ? 2 * cap : 64; fr = realloc(fr, (size_t)cap * sizeof *fr); if (!fr) return 1; }
    fr[nf++] = f;
  }
  fclose(fl);
  if (!nf) { printf("%s lists no frame\n", list); return 1; }
  static rx_t rx;
  if (rx_init(&rx)) { printf("focus_any_init failed\n"); return 1; }
  dig_t total = DIG0;
  int blocks = 0, registered = 0, held = 0;
  for (int i = 0; i < nf; i++) {
    out_t o;
    decode_one(&rx, fr[i].px, fr[i].w, fr[i].h, fr[i].held >= 0 ? fr[i].held : held, 1, &o);
    if (o.has_word) held = o.version;
    total = out_line("frame", i, fr[i].name, &o, total);
    blocks += o.got; registered += o.ring >= 0;
  }
  printf("total %d frames, %d registered, %d blocks, digest %016llx\n", nf, registered, blocks, (unsigned long long)total);
  // ai: Timing: the list again, nothing hashed, each frame's decode alone inside the clock.
  double *per = passes > 0 ? malloc((size_t)passes * sizeof(double)) : NULL;
  for (int p = 0; p < passes && per; p++) {
    double ms = 0, det = 0, smp = 0, dec = 0;
    int got = 0;
    held = 0;
    for (int i = 0; i < nf; i++) {
      out_t o;
      const double t0 = now_ms();
      decode_one(&rx, fr[i].px, fr[i].w, fr[i].h, fr[i].held >= 0 ? fr[i].held : held, 0, &o);
      ms += now_ms() - t0;
      if (o.has_word) held = o.version;
      got += o.got; det += o.ms_detect; smp += o.ms_sample; dec += o.ms_decode;
    }
    per[p] = ms / nf;
    int cpu, mhz;
    where(&cpu, &mhz);
    printf("time pass %d: %.3f ms a frame (register %.3f, sample and transform %.3f, soft values and LDPC %.3f), %d blocks, ended on cpu %d at %d MHz%s\n", p, ms / nf, det / nf, smp / nf, dec / nf, got, cpu, mhz, got == blocks ? "" : "  (NOT the digest pass's count)");
  }
  if (per) {
    for (int i = 0; i < passes; i++) for (int j = i + 1; j < passes; j++) if (per[j] < per[i]) { const double t = per[i]; per[i] = per[j]; per[j] = t; }
    printf("time median of %d passes: %.3f ms a frame (%s), fastest %.3f, slowest %.3f, %d frames\n", passes, passes & 1 ? per[passes / 2] : 0.5 * (per[passes / 2 - 1] + per[passes / 2]), BUILD_VECTOR, per[0], per[passes - 1], nf);
    free(per);
  }
  for (int i = 0; i < nf; i++) free(fr[i].px);
  free(fr); rx_free(&rx);
  return 0;
}

// ai: ---------------------------------------------------------------- synth

static float rnd_f(void) { return (float)((int)(rnd() & 0xffff) - 32768) / 32768.0f; }

// ai: The transform's kernels on every shape a stage takes: each radix, sizes on the radix-2 schedule and with the
// ai: radix-3 stage, column counts that leave each remainder of a vector of four, both directions, a pitch wider
// ai: than the block.
static void synth_fft(void) {
  static const int ROWS[] = { 2, 4, 8, 12, 64, 96, 128, 256, 384, 512, 768, 1024, 1536 }, COLS[] = { 1, 2, 3, 4, 5, 7, 8, 32, 61, 64 };
  for (int radix = 2; radix <= 4; radix += 2) {
    fft_set_radix(radix);
    for (size_t r = 0; r < sizeof ROWS / sizeof *ROWS; r++) {
      const int rows = ROWS[r];
      dig_t d = DIG0;
      for (size_t c = 0; c < sizeof COLS / sizeof *COLS; c++) for (int inv = 0; inv < 2; inv++) {
        const int cols = COLS[c], pitch = cols + 3;
        float *re = malloc((size_t)rows * pitch * sizeof(float)), *im = malloc((size_t)rows * pitch * sizeof(float));
        rnd_seed((uint64_t)(rows * 131 + cols * 7 + inv));
        for (int i = 0; i < rows * pitch; i++) { re[i] = rnd_f(); im[i] = rnd_f(); }
        fft_cols(re, im, rows, cols, pitch, inv);
        d = dig(d, re, (size_t)rows * pitch * sizeof(float)); d = dig(d, im, (size_t)rows * pitch * sizeof(float));
        free(re); free(im);
      }
      printf("synth fft radix %d rows %4d: %016llx\n", radix, rows, (unsigned long long)d);
    }
    for (int n = 64; n <= 96; n += 32) {
      float *re = malloc((size_t)n * n * sizeof(float)), *im = malloc((size_t)n * n * sizeof(float));
      rnd_seed((uint64_t)n);
      for (int i = 0; i < n * n; i++) { re[i] = rnd_f(); im[i] = rnd_f(); }
      fft2d(re, im, n, 0);
      dig_t d = dig(dig(DIG0, re, (size_t)n * n * sizeof(float)), im, (size_t)n * n * sizeof(float));
      fft2d(re, im, n, 1);
      d = dig(dig(d, re, (size_t)n * n * sizeof(float)), im, (size_t)n * n * sizeof(float));
      printf("synth fft2d radix %d n %d: %016llx\n", radix, n, (unsigned long long)d);
      free(re); free(im);
    }
  }
  fft_set_radix(2);   // ai: the reference's, for everything after
}

// ai: RGBA to luma at every remainder the sixteen-pixel loop leaves, against the formula too.
static int synth_luma(void) {
  dig_t d = DIG0;
  int bad = 0;
  for (int k = 0; k <= 132; k++) {
    const int n = k < 132 ? k : 100003;
    uint8_t *px = malloc((size_t)4 * n + 4), *out = malloc((size_t)n + 1);
    rnd_seed((uint64_t)n + 7);
    for (int i = 0; i < 4 * n; i++) px[i] = (uint8_t)rnd();
    ob_luma(px, n, out);
    for (int i = 0; i < n; i++) bad += out[i] != (uint8_t)((77 * px[4 * i] + 150 * px[4 * i + 1] + 29 * px[4 * i + 2] + 128) >> 8);
    d = dig(d, out, (size_t)n);
    free(px); free(out);
  }
  printf("synth luma: %016llx, %d pixels off the formula\n", (unsigned long long)d, bad);
  return bad != 0;
}

static void synth_binarize(void) {
  static const int SZ[][2] = { { 16, 16 }, { 37, 29 }, { 64, 8 }, { 129, 131 }, { 640, 480 }, { 1283, 961 }, { 1080, 1080 } };
  dig_t d = DIG0;
  for (size_t s = 0; s < sizeof SZ / sizeof *SZ; s++) for (int kind = 0; kind < 2; kind++) {
    const int w = SZ[s][0], h = SZ[s][1];
    uint8_t *img = malloc((size_t)w * h + 64), *bin = malloc((size_t)w * h + 64);
    rnd_seed((uint64_t)(w * 3 + h + kind));
    for (int i = 0; i < w * h; i++) img[i] = (uint8_t)(kind ? (((i % w) >> 2) & 1 ? 30 + (rnd() & 15) : 220 - (rnd() & 15)) : rnd());
    ob_test_binarize(img, w, h, bin);
    d = dig(d, bin, (size_t)w * h);
    free(img); free(bin);
  }
  printf("synth binarize: %016llx\n", (unsigned long long)d);
}

// ai: The interleaved syndromes: codeword counts either side of a vector of sixteen, the short last group included.
static void synth_rs(void) {
  static const int MS[] = { 1, 2, 15, 16, 17, 31, 32, 33, 40 }, NS[] = { 8, 80, 255 }, RS[] = { 3, 16, 32 };
  dig_t d = DIG0;
  for (size_t a = 0; a < sizeof MS / sizeof *MS; a++) for (size_t b = 0; b < sizeof NS / sizeof *NS; b++) for (size_t c = 0; c < sizeof RS / sizeof *RS; c++) {
    const int m = MS[a], n = NS[b], nroots = RS[c];
    if (nroots >= n) continue;
    uint8_t *buf = malloc((size_t)m * n), *S = calloc((size_t)m * nroots, 1);
    rnd_seed((uint64_t)(m * 1000 + n * 10 + nroots));
    for (int i = 0; i < m * n; i++) buf[i] = (uint8_t)rnd();
    d = dig_int(d, rs_syndromes_il(buf, m, n, nroots, S)); d = dig(d, S, (size_t)m * nroots);
    // ai: and real codewords, one of them damaged: all syndromes zero but that one's
    uint8_t cw[255];
    for (int k = 0; k < m; k++) { for (int i = 0; i < n - nroots; i++) cw[i] = (uint8_t)rnd(); rs_encode(cw, n - nroots, nroots); for (int i = 0; i < n; i++) buf[i * m + k] = cw[i]; }
    buf[(n / 2) * m + m / 2] ^= 0x5a;
    memset(S, 0, (size_t)m * nroots);
    d = dig_int(d, rs_syndromes_il(buf, m, n, nroots, S)); d = dig(d, S, (size_t)m * nroots);
    free(buf); free(S);
  }
  printf("synth rs syndromes: %016llx\n", (unsigned long long)d);
}

// ai: A symbol painted by the codec (focus_encode, focus_paint_rgba, then the painter's RGBA back to luma) on a
// ai: white page, P pixels of it all round.
static uint8_t *paint_page(const focus_t *f, const uint8_t *blocks, int P, int *side, dig_t *d) {
  const int W = f->px + 2 * FOCUS_QUIET * f->pxm, S = W + 2 * P;
  float *drive = malloc((size_t)f->px * f->px * sizeof(float));
  uint8_t *rgba = malloc((size_t)W * W * 4), *luma = malloc((size_t)W * W), *img = malloc((size_t)S * S);
  if (!drive || !rgba || !luma || !img) { free(drive); free(rgba); free(luma); free(img); return NULL; }
  focus_encode(f, blocks, drive);
  *d = dig(*d, drive, (size_t)f->px * f->px * sizeof(float));
  focus_paint_rgba(f, drive, rgba);
  *d = dig(*d, rgba, (size_t)W * W * 4);
  ob_luma(rgba, W * W, luma);
  *d = dig(*d, luma, (size_t)W * W);
  memset(img, 255, (size_t)S * S);
  for (int y = 0; y < W; y++) memcpy(img + (size_t)(y + P) * S + P, luma + (size_t)y * W, (size_t)W);
  free(drive); free(rgba); free(luma);
  *side = S;
  return img;
}

// ai: Every ring with every picture size: painted, read by the blind receiver, every block back byte for byte. The
// ai: pairs and their payload are scripts/exp/ring_pairs.mjs's, the gate that holds the same on wasm.
static int synth_pairs(void) {
  static const int SUBCH[] = { 16, 48, 96, 320, 336, 1024 };   // ai: 256, 384, 512, 768, 1024, 1536
  static rx_t rx;
  int fails = 0, idx = 0;
  dig_t total = DIG0;
  if (rx_init(&rx)) { printf("focus_any_init failed\n"); return 1; }
  for (int r = 0; r < FOCUS_RINGS; r++) for (size_t s = 0; s < sizeof SUBCH / sizeof *SUBCH; s++) {
    const int subch = SUBCH[s], n = focus_n_for(subch);
    focus_t f;
    if (focus_init(&f, n, subch, FOCUS_LDPC, 2, 2 * FOCUS_RING[r], 0, 0, 0, 0, 0, 0, 0)) { printf("focus_init %d in ring %d failed\n", subch, FOCUS_RING[r]); fails++; continue; }
    uint8_t *blocks = malloc((size_t)f.blocks * f.block_bytes);
    for (int i = 0; i < f.blocks * f.block_bytes; i++) blocks[i] = (uint8_t)((uint32_t)((uint64_t)i * 2654435761ULL + (uint64_t)(r * 97 + subch)) >> 13);   // ai: scripts/exp/ring_pairs.mjs's payload
    dig_t enc = DIG0;
    int S = 0;
    uint8_t *img = paint_page(&f, blocks, 24, &S, &enc);
    if (!img) { printf("out of memory\n"); return 1; }
    out_t o;
    decode_one(&rx, img, S, S, 0, 1, &o);
    int back = 0;
    for (int b = 0; b < f.blocks; b++) back += rx.ok[b] && !memcmp(rx.blocks + (size_t)b * rx.B, blocks + (size_t)b * f.block_bytes, (size_t)f.block_bytes);
    const int good = o.ring == r && o.n == n && o.has_word && o.version == subch / FOCUS_GROUP && back == f.blocks;
    if (!good) fails++;
    char name[64];
    snprintf(name, sizeof name, "ring%d_n%d_%dpx", FOCUS_RING[r], n, S);
    printf("synth pair ring %3d LIZARD-%-4d n %4d: encode %016llx, %d of %d blocks back%s\n", FOCUS_RING[r], subch, n, (unsigned long long)enc, back, f.blocks, good ? "" : "  FAILED");
    total = dig(total, &enc, sizeof enc);
    total = out_line("synth", idx++, name, &o, total);
    free(img); free(blocks); focus_free(&f);
  }
  printf("synth pairs: %d of %d read blind, digest %016llx\n", idx - fails, idx, (unsigned long long)total);
  rx_free(&rx);
  return fails;
}

// ai: The lab shapes scripts/exp/simd_check.mjs holds on wasm, told: a small picture, the paper's hard-decision RS, a power
// ai: tilt, tiers at three code rates, and the resampled ring and picture pairs.
static int synth_lab(void) {
  static const struct { int n, subch, mode, span; float tilt; int tiers; } LAB[] = {
    { 128, 8, FOCUS_LDPC, 128, 0, 0 }, { 256, 48, FOCUS_LDPC, 256, 0, 0 }, { 1024, 192, FOCUS_LDPC, 256, 0, 0 }, { 512, 64, FOCUS_RS, 256, 0, 0 },
    { 512, 320, FOCUS_LDPC, 256, 9, 0 }, { 512, 0, FOCUS_LDPC, 256, 9, 1 }, { 384, 64, FOCUS_LDPC, 256, 0, 0 }, { 512, 96, FOCUS_LDPC, 192, 0, 0 }, { 768, 320, FOCUS_LDPC, 128, 0, 0 } };
  static const focus_tier_t TIERS[3] = { { 6, 8, 7 }, { 4, 12, 8 }, { 2, 12, 12 } };
  int fails = 0;
  for (size_t k = 0; k < sizeof LAB / sizeof *LAB; k++) {
    focus_t f;
    const int bad = LAB[k].tiers ? focus_init_tiers(&f, LAB[k].n, TIERS, 3, 2, LAB[k].span, LAB[k].tilt, 0, 0, 0, 0, 0, 0)
                                 : focus_init(&f, LAB[k].n, LAB[k].subch, LAB[k].mode, 2, LAB[k].span, LAB[k].tilt, 0, 0, 0, 0, 0, 0);
    if (bad) { printf("synth lab %zu: init failed\n", k); fails++; continue; }
    uint8_t *blocks = malloc((size_t)f.blocks * f.block_bytes), *got = calloc((size_t)f.blocks * f.block_bytes, 1), *ok = calloc((size_t)f.blocks, 1);
    rnd_seed(900 + k);
    for (int i = 0; i < f.blocks * f.block_bytes; i++) blocks[i] = (uint8_t)rnd();
    dig_t d = DIG0;
    int S = 0;
    uint8_t *img = paint_page(&f, blocks, 24, &S, &d);
    if (!img) { printf("out of memory\n"); return 1; }
    ob_result_t res;
    ob_debug_hash = 1;
    const int n = focus_decode(&f, img, S, S, 1.0f, 2, got, ok, &res);
    const uint32_t spec = ob_debug_hash; ob_debug_hash = 0;
    const int8_t *its; const float *est;
    focus_block_stats(&f, &its, &est);
    d = dig_int(d, n); d = dig(d, &spec, sizeof spec); d = dig(d, res.quad, sizeof res.quad); d = dig(d, &res.mark_score, sizeof res.mark_score); d = dig_int(d, res.orient);
    d = dig(d, ok, (size_t)f.blocks); d = dig(d, got, (size_t)f.blocks * f.block_bytes);
    if (f.mode == FOCUS_LDPC) { d = dig(d, its, (size_t)f.blocks); d = dig(d, est, (size_t)f.blocks * sizeof(float)); }
    const int back = n == f.blocks && !memcmp(got, blocks, (size_t)f.blocks * f.block_bytes);
    if (!back) fails++;
    printf("synth lab n %4d subch %3d mode %d span %3d tilt %g%s: %016llx, %d of %d blocks%s\n", LAB[k].n, f.subch, LAB[k].mode, LAB[k].span, (double)LAB[k].tilt, LAB[k].tiers ? " tiers" : "", (unsigned long long)d, n, f.blocks, back ? "" : "  FAILED");
    free(img); free(blocks); free(got); free(ok); focus_free(&f);
  }
  return fails;
}

// ai: The binary grid code (no further work, but its demapper and finder carry vector paths that must run): modules
// ai: four pixels wide, blurred, shaded and noised in integers, read under the option sets scripts/exp/simd_check.mjs uses.
static int synth_binary(void) {
  static const struct { int w, tiles, rate; } LAY[] = { { 320, 3, 4 }, { 256, 2, 3 } };
  static const struct { const char *name; float gamma; int eq, taps, regions, mesh, iters, pilot; } OPT[] = {
    { "blind 3x3", 1, OB_EQ_BLIND, 3, 1, 1, 50, 0 }, { "blind 5x5, 3x3 regions", 1, OB_EQ_BLIND, 5, 3, 1, 50, 0 },
    { "pilot 5x5", 1, OB_EQ_PILOT, 5, 1, 1, 50, 8 }, { "no equalizer, no mesh, gamma 2.2", 2.2f, OB_EQ_NONE, 3, 1, 0, 30, 0 } };
  int fails = 0;
  for (size_t l = 0; l < sizeof LAY / sizeof *LAY; l++) for (size_t q = 0; q < sizeof OPT / sizeof *OPT; q++) {
    ob_layout_t L;
    const ob_cfg_t cfg = { .w = LAY[l].w, .h = LAY[l].w, .tx = LAY[l].tiles, .ty = LAY[l].tiles, .rate = LAY[l].rate, .map = 0, .pilot_step = OPT[q].pilot };
    if (ob_layout_init(&L, &cfg)) { printf("synth binary: layout failed\n"); fails++; continue; }
    const int K = 4, P = 40, W = L.w * K + 2 * P, H = L.h * K + 2 * P, B = L.block_bytes;
    uint8_t *blocks = malloc((size_t)L.tiles * B), *got = calloc((size_t)L.tiles * B, 1), *mod = malloc((size_t)L.w * L.h), *a = malloc((size_t)W * H), *img = malloc((size_t)W * H);
    rnd_seed(500 + l * 10 + q);
    for (int i = 0; i < L.tiles * B; i++) blocks[i] = (uint8_t)rnd();
    ob_encode(&L, blocks, mod);
    memset(a, 225, (size_t)W * H);
    for (int y = 0; y < L.h * K; y++) for (int x = 0; x < L.w * K; x++) a[(size_t)(y + P) * W + x + P] = mod[(y / K) * L.w + x / K] ? 35 : 225;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      int s = 0;
      for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { const int yy = clampi(y + dy, 0, H - 1), xx = clampi(x + dx, 0, W - 1); s += a[(size_t)yy * W + xx]; }
      int v = (s + 4) / 9;
      v = v * (1024 - 180 * y / H) >> 10;
      v += (int)(rnd() % 9) - 4;
      img[(size_t)y * W + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    ob_opts_t o;
    ob_default_opts(&o);
    o.gamma = OPT[q].gamma; o.eq_mode = OPT[q].eq; o.eq_taps = OPT[q].taps; o.eq_regions = OPT[q].regions; o.mesh = OPT[q].mesh; o.max_iter = OPT[q].iters; o.llr_gain = 0.7f; o.llr_clip = 10;
    ob_result_t res;
    ob_debug_hash = 1;
    const int n = ob_decode(&L, img, W, H, &o, got, &res, NULL);
    const uint32_t hsh = ob_debug_hash; ob_debug_hash = 0;
    dig_t d = DIG0;
    d = dig_int(d, n); d = dig(d, &hsh, sizeof hsh); d = dig_int(d, res.found); d = dig_int(d, res.finders); d = dig(d, res.quad, sizeof res.quad); d = dig_int(d, res.orient);
    d = dig(d, &res.mark_score, sizeof res.mark_score); d = dig(d, res.ok, sizeof res.ok); d = dig(d, res.iters, sizeof res.iters); d = dig(d, got, (size_t)L.tiles * B);
    int back = 0;
    for (int t = 0; t < L.tiles; t++) back += res.ok[t] && !memcmp(got + (size_t)t * B, blocks + (size_t)t * B, (size_t)B);
    if (back != L.tiles) fails++;
    printf("synth binary %d modules, %s: %016llx, %d of %d tiles back%s\n", L.w, OPT[q].name, (unsigned long long)d, back, L.tiles, back == L.tiles ? "" : "  FAILED");
    free(blocks); free(got); free(mod); free(a); free(img); ob_layout_free(&L);
  }
  return fails;
}

static int synth(void) {
  int fails = 0;
  synth_fft();
  fails += synth_luma();
  synth_binarize();
  synth_rs();
  fails += synth_lab();
  fails += synth_binary();
  fails += synth_pairs();
  printf("synth: %s\n", fails ? "FAILED" : "every round trip came back");
  return fails != 0;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IOLBF, 0);   // ai: a line at a time, so a run that dies has printed every frame before the one it died on
  const int fused = contracts();
  printf("build %s, %s, contraction %s\n", BUILD_VECTOR, __VERSION__, fused ? "ON: this build fuses a multiply and an add (-ffp-contract=off is missing)" : "off");
  if (fused) return 1;
  if (argc >= 2 && !strcmp(argv[1], "unit")) return unit(argc >= 3 && !strcmp(argv[2], "full"));
  if (argc >= 2 && !strcmp(argv[1], "synth")) return synth();
  if (argc >= 3 && !strcmp(argv[1], "frames")) return frames(argv[2], argc >= 4 ? atoi(argv[3]) : 0);
  printf("usage: neon_check unit [full] | synth | frames <list> [passes]\n  a list line: w h held path   (raw 8-bit luma; held -1: the last word read in the list)\n");
  return 2;
}
