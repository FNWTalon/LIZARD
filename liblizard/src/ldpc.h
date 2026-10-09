// Quasi-cyclic IRA LDPC: H = [Hd | Hp], Hd built from Z x Z circulants over a base graph of 48 columns
// (49 for the format's 7/8 code, whose first data column is never sent: np below), Hp the bit-level
// accumulator staircase. The format's codes ship as their base matrices (ldpc_base.h, one a rate of the
// rate profile, each with its own kb, mb and np); any other is derived from (n, rate, seed) by
// ldpc_generate. The construction is pre-2005 literature throughout (Gallager codes, IRA staircase
// encoding, circulant lifting with girth conditioning).
#ifndef OB_LDPC_H
#define OB_LDPC_H
#include <stdint.h>

enum { LDPC_NB = 48, LDPC_RATES = 7 };

typedef struct {
  int n, k, m, z, kb, mb, rate, norm;
  int np, nt;         // codeword bits never sent (the first np, whole data columns), and n - np, what a block's slots carry
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

// The code at z = n_max / 48 and the given rate index: the format's table where one matches (z, rate, seed 1),
// else generated (48 columns, no puncturing). Returns 0 on success.
int ldpc_init(ldpc_t *c, int n_max, int rate, uint32_t seed);
// Which set of the format's tables ldpc_init builds from: 0 today's (ldpc_base.h LDPC_BASES), 1 the set before
// 2026-10-08 (LDPC_BASES_V1, what the recordings of 2026-10-07 and 08 were painted with), 2 a lab set (LDPC_BASES_V2:
// the 2/3 and 1/2 codes at z = 128, for timing the GPU's kernels; not the format); -1 (the default) reads
// LIZ_TABLES from the environment once, 0 where there is none. Set before the codecs are built.
extern int ldpc_codes_set;
// The generator: shift[r][j] for the mb block rows of the code at circulant size z (the profile row, the seed's
// xorshift draws, the cycle score), -1 where there is no edge. Returns mb, or -1. It builds every code but the
// format's four, which are the tables they were once generated into; test/ldpc_table.c holds each equal to it.
int ldpc_generate(int z, int rate, uint32_t seed, int shift[][LDPC_NB]);
void ldpc_free(ldpc_t *c);

// bits: one bit per byte. data k in, codeword n out (systematic, data first).
void ldpc_encode(const ldpc_t *c, const uint8_t *data, uint8_t *codeword);

// llr > 0 means bit 0; the np bits never sent are fed as 0. Returns iterations used, or -1 if the syndrome never
// cleared. out receives the n hard decisions either way.
int ldpc_decode(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter);
// The same with a way out for a block that is going nowhere (ldpc.c). Returns -3 when it took it.
int ldpc_decode_stall(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter, int stall_it, float stall_ratio);
// The same code decoded one check at a time. Kept as the reference the block-row schedule is measured against (test/ldpc_sched.c).
int ldpc_decode_serial(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter);

// The code's shape (12 ints: n, k, m, z, zp, mb, kb, norm, slots_max, slots_total, slots, np) and its block-row
// tables, for a decoder running somewhere else. See ldpc.c.
void ldpc_tables(const ldpc_t *c, int32_t *dims, int32_t *lay);

#endif
