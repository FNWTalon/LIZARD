// ai: ob_crc32c, the block's CRC-32C (src/layout.c; ARMv8's CRC32C instructions where arm64 has them, SSE4.2's crc32 on
// ai: x86-64) against the bit loop: 200,000 random buffers of 0 to 600 bytes at random offsets (every tail short of 8),
// ai: every value compared, and the standard check value; then a 476-byte block's time each way. Exit 1 on any
// ai: difference or a wrong check value. Any platform; test/neon_kernels.sh runs it on a phone; on x86-64, from
// ai: liblizard/: gcc -O3 -ffp-contract=off -DNDEBUG -msse4.1 -Wall -Wextra -Isrc -o build/crc_check test/crc_check.c
// ai: src/layout.c src/fmt.c src/rs.c && build/crc_check
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
uint32_t ob_crc32c(const uint8_t *p, int n);
static uint32_t loop(const uint8_t *p, int n) { uint32_t c = 0xffffffffu; for (int i = 0; i < n; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82f63b78u & (0u - (c & 1))); } return ~c; }
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
int main(void) {
  static uint8_t buf[1024]; uint32_t z = 5; int bad = 0;
  for (int t = 0; t < 200000; t++) {
    const int n = (int)(z % 601), o = (int)((z >> 10) % 16);
    for (int i = 0; i < n + 16; i++) { z = z * 1664525u + 1013904223u; buf[i] = (uint8_t)(z >> 24); }
    if (ob_crc32c(buf + o, n) != loop(buf + o, n)) bad++;
  }
  const uint32_t check = ob_crc32c((const uint8_t *)"123456789", 9);
  if (check != 0xe3069283u) bad++;
  printf("CRC-32C: 200000 buffers, %d apart (\"123456789\": %08x, the standard check value e3069283)\n", bad, check);
  double ta = 1e30, tb = 1e30; volatile uint32_t sink = 0;
  for (int r = 0; r < 50; r++) {
    double t0 = now_ns(); for (int k = 0; k < 100; k++) sink += loop(buf, 476); double t = now_ns() - t0; if (t < ta) ta = t;
    t0 = now_ns(); for (int k = 0; k < 100; k++) sink += ob_crc32c(buf, 476); t = now_ns() - t0; if (t < tb) tb = t;
  }
  printf("a 476-byte block: the bit loop %.0f ns, ob_crc32c %.0f ns (%.0fx)\n", ta / 100, tb / 100, ta / tb);
  return bad != 0;
}
