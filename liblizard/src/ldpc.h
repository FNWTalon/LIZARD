// Quasi-cyclic IRA LDPC: H = [Hd | Hp], Hd built from Z x Z circulants over a base graph of kb data columns and mb
// block rows (each of the format's four codes its own: ldpc_base.h, z = 128 for every one), Hp the bit-level
// accumulator staircase. The format's codes ship as their base matrices, one a rate of the rate profile; the 7/8
// code's last data column, the zeros past the block, is never sent (np below). The construction is pre-2005 literature throughout (Gallager
// codes, IRA staircase encoding, circulant lifting with girth conditioning).
#ifndef OB_LDPC_H
#define OB_LDPC_H
#include <stdint.h>

enum { LDPC_RATES = 7 };

typedef struct {
  int n, k, m, z, kb, mb, rate, norm;
  int np, nt;         // codeword bits never sent (the last np of the data, whole data columns: the 7/8 code's known zeros), and n - np, what a block's slots carry
  int edges;          // entries in H
  int *row_ptr;       // m + 1
  int *col_idx;       // edges, variable index per entry, row-major
  // Block-row view for the decoder: the data circulants of each block row, and its workspace.
  int slots_max, slots_total, zp;   // most entries in a block row (staircase included), their sum, z rounded up to 8
  int *lay_ptr, *lay_col, *lay_shift;
  int16_t *wl, *wq;   // posteriors (data in code order, parity transposed to [block row][position]); one layer's gathered rows
  int8_t *wr;         // check-to-bit messages, [slot][position]
  int wide;           // the layer kernel ldpc_init chose: 0 the 128-bit one (or the scalar twin), 1 AVX2, 2 AVX-512; -1 set by a check:
                      // the 128-bit kernel on today's layout, where arm64 or x86-64 would take its own. With w4 on x86-64, the
                      // four-times kernel's form: 0 SSE4.1, 1 AVX2
  int16_t *w4;        // arm64 (OB_NEON) and x86-64 (OB_X86), z a multiple of 32: the data columns four times over, kb x 4z, so the
                      // rotation of any slot is one run of z inside [z, 3z); 0 elsewhere
} ldpc_t;

// ai: Rate indices 0 to LDPC_RATES - 1, the format's four with a code: 2 (1/2), 3 (2/3), 4 (3/4), 6 (7/8); the name and
// ai: value of any other index are NULL and 0.
extern const char *const ldpc_rate_name[LDPC_RATES];
double ldpc_rate_value(int rate);

// The format's code at a rate index (ldpc_base.h, the rate profile's four). Returns 0 on success, -1 for a rate index
// with no code. test/ldpc_table.c holds each table to the hash of its H.
int ldpc_init(ldpc_t *c, int rate);
// The bits a block's slots carry under the format's code at a rate index (n - np), without building it; 0 for none.
int ldpc_sent_bits(int rate);
void ldpc_free(ldpc_t *c);

// bits: one bit per byte. data k in, codeword n out (systematic, data first).
void ldpc_encode(const ldpc_t *c, const uint8_t *data, uint8_t *codeword);

// llr > 0 means bit 0; the np bits never sent are fed as the caller knows them (the 7/8 code's zeros at 127). Returns iterations used, or -1 if the syndrome never
// cleared. out receives the n hard decisions either way.
int ldpc_decode(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter);
// The same with a way out for a block that is going nowhere (ldpc.c). Returns -3 when it took it.
int ldpc_decode_stall(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter, int stall_it, float stall_ratio);

// The code's shape (12 ints: n, k, m, z, zp, mb, kb, norm, slots_max, slots_total, slots, np) and its block-row
// tables, for a decoder running somewhere else. See ldpc.c.
void ldpc_tables(const ldpc_t *c, int32_t *dims, int32_t *lay);

#endif
