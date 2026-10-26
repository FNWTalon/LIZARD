// Reed-Solomon over GF(256), polynomial 0x11d, generator roots alpha^0 .. alpha^(nroots-1).
// Hard-decision bounded-distance decoding, for the format word (fmt.c): soft LDPC carries the payload, and RS the small
// fixed field (research/08).
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

#endif
