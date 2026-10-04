// Reed-Solomon over GF(256), polynomial 0x11d, generator roots alpha^0 .. alpha^(nroots-1).
// Hard-decision bounded-distance decoding: what FOCUS and the 2D symbologies use, kept here as
// the control against soft decoding.
#ifndef OB_RS_H
#define OB_RS_H
#include <stdint.h>

// cw: k data bytes followed by nroots parity bytes, n = k + nroots <= 255.
void rs_encode(uint8_t *cw, int k, int nroots);
// Corrects in place. Returns the number of byte errors corrected, or -1 when there are more
// than nroots / 2 (detected, not corrected).
int rs_decode(uint8_t *cw, int n, int nroots);
// The same with erasures: eras[0 .. neras) are positions the receiver distrusts (any order, no repeats).
// Succeeds when 2 * errors + erasures <= nroots. Returns errors + erasures corrected, or -1.
int rs_decode_er(uint8_t *cw, int n, int nroots, const int *eras, int neras);

// m codewords of n bytes each, interleaved byte by byte (byte j of the buffer belongs to codeword j % m,
// position j / m), which is how a block of several codewords is laid out so that a smear is shared between
// them. Syndromes of all m at once: S[c * nroots + i]. Returns 1 if any codeword has a nonzero syndrome.
// With wasm SIMD the codewords are the lanes: every lane multiplies by the same constant alpha^i, which is
// two 16-entry table lookups (low and high nibble) for sixteen codewords at a time.
int rs_syndromes_il(const uint8_t *buf, int m, int n, int nroots, uint8_t *S);
// Decode codeword c of an interleaved buffer in place, from syndromes already computed (errors and erasures).
int rs_decode_il(uint8_t *buf, int m, int n, int nroots, int c, const uint8_t *S, const int *eras, int neras);

#endif
