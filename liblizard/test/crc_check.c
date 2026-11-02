// ai: ob_crc32 (src/layout.c; ARMv8's CRC32 instructions where arm64 has them) against the bit loop: 200,000 random
// ai: buffers of 0 to 600 bytes at random offsets, every value compared, and the standard check value; then a 476-byte
// ai: block's time each way. Any platform; test/neon_kernels.sh runs it on a phone: crc_check
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
uint32_t ob_crc32(const uint8_t *p, int n);
static uint32_t loop(const uint8_t *p, int n) { uint32_t c = 0xffffffffu; for (int i = 0; i < n; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1))); } return ~c; }
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
int main(void) {
  static uint8_t buf[1024]; uint32_t z = 5; int bad = 0;
  for (int t = 0; t < 200000; t++) {
    const int n = (int)(z % 601), o = (int)((z >> 10) % 16);
    for (int i = 0; i < n + 16; i++) { z = z * 1664525u + 1013904223u; buf[i] = (uint8_t)(z >> 24); }
    if (ob_crc32(buf + o, n) != loop(buf + o, n)) bad++;
  }
  printf("CRC-32: 200000 buffers, %d apart (\"123456789\": %08x, the standard check value cbf43926)\n", bad, ob_crc32((const uint8_t *)"123456789", 9));
  double ta = 1e30, tb = 1e30; volatile uint32_t sink = 0;
  for (int r = 0; r < 50; r++) {
    double t0 = now_ns(); for (int k = 0; k < 100; k++) sink += loop(buf, 476); double t = now_ns() - t0; if (t < ta) ta = t;
    t0 = now_ns(); for (int k = 0; k < 100; k++) sink += ob_crc32(buf, 476); t = now_ns() - t0; if (t < tb) tb = t;
  }
  printf("a 476-byte block: the bit loop %.0f ns, the CRC32 instructions %.0f ns (%.0fx)\n", ta / 100, tb / 100, ta / tb);
  return bad != 0;
}
