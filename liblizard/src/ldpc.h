// Quasi-cyclic IRA LDPC: H = [Hd | Hp], Hd built from Z x Z circulants over a 48-column base
// graph, Hp the bit-level accumulator staircase. The format's code ships as its base matrix
// (ldpc_base.h); any other is derived from (n, rate, seed) by ldpc_generate. The construction is
// pre-2005 literature throughout (Gallager codes, IRA staircase encoding, circulant lifting with
// girth conditioning).
#ifndef OB_LDPC_H
#define OB_LDPC_H
#include <stdint.h>

enum { LDPC_NB = 48, LDPC_RATES = 7 };

typedef struct {
  int n, k, m, z, kb, mb, rate, norm;
  int edges;          // entries in H
  int *row_ptr;       // m + 1
  int *col_idx;       // edges, variable index per entry, row-major
  // Block-row view for the decoder: the data circulants of each block row, and its workspace.
  int slots_max, slots_total, zp;   // most entries in a block row (staircase included), their sum, z rounded up to 8
  int *lay_ptr, *lay_col, *lay_shift;
  int16_t *wl, *wq;   // posteriors (data in code order, parity transposed to [block row][position]); one layer's gathered rows
  int8_t *wr;         // check-to-bit messages, [slot][position]
} ldpc_t;

extern const char *const ldpc_rate_name[LDPC_RATES];
double ldpc_rate_value(int rate);

// Largest code with n <= n_max at the given rate index. Returns 0 on success.
int ldpc_init(ldpc_t *c, int n_max, int rate, uint32_t seed);
// The generator: shift[r][j] for the mb block rows of the code at circulant size z (the profile row, the seed's
// xorshift draws, the cycle score), -1 where there is no edge. Returns mb, or -1. It builds every code but the
// format's, which is the table it was once generated into; test/ldpc_table.c holds the two equal.
int ldpc_generate(int z, int rate, uint32_t seed, int shift[][LDPC_NB]);
void ldpc_free(ldpc_t *c);

// bits: one bit per byte. data k in, codeword n out (systematic, data first).
void ldpc_encode(const ldpc_t *c, const uint8_t *data, uint8_t *codeword);

// llr > 0 means bit 0. Returns iterations used, or -1 if the syndrome never cleared.
// out receives the n hard decisions either way.
int ldpc_decode(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter);
// The same with a way out for a block that is going nowhere (ldpc.c). Returns -3 when it took it.
int ldpc_decode_stall(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter, int stall_it, float stall_ratio);
// The same code decoded one check at a time. Kept as the reference the block-row schedule is measured against (test/ldpc_sched.c).
int ldpc_decode_serial(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter);

// The code's shape and its block-row tables, for a decoder running somewhere else. See ldpc.c.
void ldpc_tables(const ldpc_t *c, int32_t *dims, int32_t *lay);

#endif
