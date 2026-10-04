// SHAKE256 (FIPS 202), for the test stream's block contents.
//
// The stream has to be a function of the block id alone, so a receiver can check any block it
// decodes without being told which frame carried it. A keyed generator would need the key on
// both ends; a hash of the id needs nothing.
#ifndef SHAKE_H
#define SHAKE_H

#include <stdint.h>
#include <stddef.h>

// SHAKE256(in) -> out. One shot: absorb all of `in`, squeeze `out_len` bytes.
void shake256(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);

// The test stream: SHAKE256 of the block id as four little-endian bytes.
//
// Why a hash and not a PRNG walk: xorshift32 has a single orbit of 2^32-1, so two ids whose
// seeds land within one block of each other on that orbit emit byte strings that are shifts of
// one another (measured at the birthday rate, ~0.6% of blocks over 40k ids), and the one id
// that seeds zero emits an all-zero block. Neither can happen here.
void stream_fill(uint32_t id, uint8_t *out, uint32_t len);

#endif
