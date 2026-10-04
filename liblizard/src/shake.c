// SHAKE256 (FIPS 202). See shake.h for why the test stream hashes its block id.
#include "shake.h"

#include <string.h>

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

static const uint64_t RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
  0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
  0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

// rho offsets in the same y*5+x order as the state.
static const int RHO[25] = {
   0,  1, 62, 28, 27,
  36, 44,  6, 55, 20,
   3, 10, 43, 25, 39,
  41, 45, 15, 21,  8,
  18,  2, 61, 56, 14,
};

static void keccak_f1600(uint64_t a[25]) {
  for (int round = 0; round < 24; round++) {
    uint64_t c[5], d[5], b[25];

    // theta
    for (int x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
    for (int x = 0; x < 5; x++) d[x] = c[(x + 4) % 5] ^ ROTL64(c[(x + 1) % 5], 1);
    for (int y = 0; y < 5; y++) for (int x = 0; x < 5; x++) a[y * 5 + x] ^= d[x];

    // rho and pi. Lanes are flat as x + 5y, so B[y][(2x+3y) % 5] lands at y + 5*((2x+3y) % 5).
    for (int y = 0; y < 5; y++) for (int x = 0; x < 5; x++) {
      const int i = y * 5 + x;
      b[y + 5 * ((2 * x + 3 * y) % 5)] = RHO[i] ? ROTL64(a[i], RHO[i]) : a[i];
    }

    // chi
    for (int y = 0; y < 5; y++) for (int x = 0; x < 5; x++)
      a[y * 5 + x] = b[y * 5 + x] ^ (~b[y * 5 + (x + 1) % 5] & b[y * 5 + (x + 2) % 5]);

    // iota
    a[0] ^= RC[round];
  }
}

// SHAKE256: capacity 512 bits, so a rate of 1600 - 512 = 1088 bits = 136 bytes.
#define RATE 136

static void absorb_lane(uint64_t a[25], size_t byte, uint8_t v) {
  a[byte / 8] ^= (uint64_t)v << (8 * (byte % 8));
}

void shake256(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len) {
  uint64_t a[25];
  memset(a, 0, sizeof a);

  size_t i = 0;
  while (in_len - i >= RATE) {
    for (size_t j = 0; j < RATE; j++) absorb_lane(a, j, in[i + j]);
    keccak_f1600(a);
    i += RATE;
  }

  // pad10*1 with the 0x1f domain separator SHAKE uses (FIPS 202 section 6.3).
  const size_t tail = in_len - i;
  for (size_t j = 0; j < tail; j++) absorb_lane(a, j, in[i + j]);
  absorb_lane(a, tail, 0x1f);
  absorb_lane(a, RATE - 1, 0x80);

  size_t done = 0;
  while (done < out_len) {
    keccak_f1600(a);
    const size_t n = out_len - done < RATE ? out_len - done : RATE;
    for (size_t j = 0; j < n; j++) out[done + j] = (uint8_t)(a[j / 8] >> (8 * (j % 8)));
    done += n;
  }
}

void stream_fill(uint32_t id, uint8_t *out, uint32_t len) {
  const uint8_t seed[4] = { (uint8_t)id, (uint8_t)(id >> 8), (uint8_t)(id >> 16), (uint8_t)(id >> 24) };
  shake256(seed, sizeof seed, out, len);
}
