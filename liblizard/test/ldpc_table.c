// The format's LDPC base matrices (src/ldpc_base.h: 7/8, 3/4 and 1/2, the rate profile's codes) against the generator
// they were written out from (src/ldpc.c ldpc_generate). Prints each table as the header holds it, then fails if the
// generator no longer makes it, or if ldpc_init's code is not the one the table describes.
//   gcc -O2 -Wall -Isrc -o build/ldpc_table test/ldpc_table.c src/ldpc.c && build/ldpc_table
// With `print` it prints the tables and checks nothing: what a deliberate change to the format's codes would paste in.
#include "ldpc.h"
#include "ldpc_base.h"
#include <stdio.h>
#include <string.h>

// FNV-1a over H's rows and columns, each code's as ldpc_init built it from the generator before its table: the 3/4
// code's 2ca2bc24 (2026-09-24), the 7/8 code's d942b2a6 and the 1/2 code's 83296c5e (2026-10-07).
static unsigned h_hash(const ldpc_t *c) {
  unsigned h = 2166136261u;
  for (int i = 0; i <= c->m; i++) h = (h ^ (unsigned)c->row_ptr[i]) * 16777619u;
  for (int e = 0; e < c->edges; e++) h = (h ^ (unsigned)c->col_idx[e]) * 16777619u;
  return h;
}
static const unsigned WANT_HASH[LDPC_BASES_N] = { 0xd942b2a6u, 0x2ca2bc24u, 0x83296c5eu };

int main(int argc, char **argv) {
  const int print = argc > 1 && !strcmp(argv[1], "print");
  int fails = 0;
  for (int t = 0; t < LDPC_BASES_N; t++) {
    const ldpc_base_t *b = &LDPC_BASES[t];
    static int gen[LDPC_NB][LDPC_NB];
    if (ldpc_generate(b->z, b->rate, LDPC_BASE_SEED, gen) != b->mb) { printf("FAIL: the generator's rate %d has not %d block rows\n", b->rate, b->mb); return 1; }
    printf("rate %s, z %d: {\n", ldpc_rate_name[b->rate], b->z);
    for (int r = 0; r < b->mb; r++) {
      printf("  {");
      for (int j = 0; j < b->kb; j++) printf(" %3d%s", gen[r][j], j + 1 < b->kb ? "," : "");
      printf(" }%s\n", r + 1 < b->mb ? "," : "");
    }
    printf("};\n");
    if (print) continue;
    for (int r = 0; r < b->mb; r++) for (int j = 0; j < LDPC_NB; j++) {
      const int want = j < b->kb ? b->shift[r * b->kb + j] : -1;
      if (gen[r][j] != want) { if (fails++ < 8) printf("FAIL: rate %s row %d column %d: the generator says %d, the table %d\n", ldpc_rate_name[b->rate], r, j, gen[r][j], want); }
    }
    // The code ldpc_init builds for the rate at its block's n_max (48 z rounded up to whole slots: the block's
    // sub-channels x 640) has to be the table's, read back from its block-row view.
    ldpc_t c;
    if (ldpc_init(&c, 48 * b->z + 47, b->rate, LDPC_BASE_SEED) || c.z != b->z || c.mb != b->mb) { printf("FAIL: ldpc_init did not build the rate %s code\n", ldpc_rate_name[b->rate]); return 1; }
    for (int r = 0; r < c.mb; r++) {
      int seen = 0;
      for (int s = c.lay_ptr[r]; s < c.lay_ptr[r + 1]; s++, seen++)
        if (b->shift[r * b->kb + c.lay_col[s]] != c.lay_shift[s]) { fails++; printf("FAIL: ldpc_init rate %s row %d column %d shift %d, the table %d\n", ldpc_rate_name[b->rate], r, c.lay_col[s], c.lay_shift[s], b->shift[r * b->kb + c.lay_col[s]]); }
      int want = 0;
      for (int j = 0; j < b->kb; j++) want += b->shift[r * b->kb + j] >= 0;
      if (seen != want) { fails++; printf("FAIL: ldpc_init rate %s row %d has %d circulants, the table %d\n", ldpc_rate_name[b->rate], r, seen, want); }
    }
    const unsigned h = h_hash(&c);
    printf("rate %s code: n %d k %d z %d, %d block rows, %d edges, H %08x%s\n", ldpc_rate_name[b->rate], c.n, c.k, c.z, c.mb, c.edges, h, h == WANT_HASH[t] ? " (as generated before the table)" : " (NOT the code generated before the table)");
    if (h != WANT_HASH[t]) fails++;
    ldpc_free(&c);
  }
  if (print) return 0;
  printf(fails ? "%d FAILED\n" : "every table is the generator's, and ldpc_init builds it\n", fails);
  return fails ? 1 : 0;
}
