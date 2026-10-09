// The format's LDPC base matrices (src/ldpc_base.h: 7/8, 3/4, 2/3 and 1/2, the rate profile's codes, and the set
// before 2026-10-08 under LIZ_TABLES=1) as ldpc_init builds them: each code's H hashed (FNV-1a over its row pointers
// and column indices) and held to the hash recorded when the table was adopted, and its block-row view held to the
// table entry by entry. Prints each table as the header holds it.
//   gcc -O2 -Wall -Isrc -o build/ldpc_table test/ldpc_table.c src/ldpc.c && build/ldpc_table
// With `print` it prints the tables and checks nothing.
#include "ldpc.h"
#include "ldpc_base.h"
#include <stdio.h>
#include <string.h>

// The 7/8 (25a3ab92), 3/4 (5660f26d) and 2/3 (12dad7a2) codes of 2026-10-08 (scripts/exp/ldpc_opt.py's tables) and the
// 1/2 code of 2026-10-07 (83296c5e, the generator's); then the set before: the generator's 7/8 (d942b2a6), 3/4
// (2ca2bc24, 2026-09-24) and 2/3 (94b14b2c).
static unsigned h_hash(const ldpc_t *c) {
  unsigned h = 2166136261u;
  for (int i = 0; i <= c->m; i++) h = (h ^ (unsigned)c->row_ptr[i]) * 16777619u;
  for (int e = 0; e < c->edges; e++) h = (h ^ (unsigned)c->col_idx[e]) * 16777619u;
  return h;
}
static const unsigned WANT_HASH[2][LDPC_BASES_N] = { { 0x25a3ab92u, 0x5660f26du, 0x12dad7a2u, 0x83296c5eu }, { 0xd942b2a6u, 0x2ca2bc24u, 0x94b14b2cu, 0x83296c5eu } };

int main(int argc, char **argv) {
  const int print = argc > 1 && !strcmp(argv[1], "print");
  int fails = 0;
  for (int set = 0; set < 2; set++) {
    const ldpc_base_t *bases = set ? LDPC_BASES_V1 : LDPC_BASES;
    ldpc_codes_set = set;
    printf("%s:\n", set ? "the set before 2026-10-08 (LIZ_TABLES=1)" : "the format's tables");
    for (int t = 0; t < LDPC_BASES_N; t++) {
      const ldpc_base_t *b = &bases[t];
      printf("rate %s, z %d, %d block rows, %d data columns, %d never sent: {\n", ldpc_rate_name[b->rate], b->z, b->mb, b->kb, b->np);
      for (int r = 0; r < b->mb; r++) {
        printf("  {");
        for (int j = 0; j < b->kb; j++) printf(" %3d%s", b->shift[r * b->kb + j], j + 1 < b->kb ? "," : "");
        printf(" }%s\n", r + 1 < b->mb ? "," : "");
      }
      printf("};\n");
      if (print) continue;
      // The code ldpc_init builds for the rate at its block's n_max (48 z rounded up to whole slots: the block's
      // sub-channels x 640) has to be the table's, read back from its block-row view.
      ldpc_t c;
      if (ldpc_init(&c, 48 * b->z + 47, b->rate, LDPC_BASE_SEED) || c.z != b->z || c.mb != b->mb || c.kb != b->kb || c.np != b->np * b->z) { printf("FAIL: ldpc_init did not build the rate %s code\n", ldpc_rate_name[b->rate]); return 1; }
      for (int r = 0; r < c.mb; r++) {
        int seen = 0;
        for (int s = c.lay_ptr[r]; s < c.lay_ptr[r + 1]; s++, seen++)
          if (b->shift[r * b->kb + c.lay_col[s]] != c.lay_shift[s]) { fails++; printf("FAIL: ldpc_init rate %s row %d column %d shift %d, the table %d\n", ldpc_rate_name[b->rate], r, c.lay_col[s], c.lay_shift[s], b->shift[r * b->kb + c.lay_col[s]]); }
        int want = 0;
        for (int j = 0; j < b->kb; j++) want += b->shift[r * b->kb + j] >= 0;
        if (seen != want) { fails++; printf("FAIL: ldpc_init rate %s row %d has %d circulants, the table %d\n", ldpc_rate_name[b->rate], r, seen, want); }
      }
      const unsigned h = h_hash(&c);
      printf("rate %s code: n %d (%d sent) k %d z %d, %d block rows, %d edges, H %08x%s\n", ldpc_rate_name[b->rate], c.n, c.nt, c.k, c.z, c.mb, c.edges, h, h == WANT_HASH[set][t] ? " (as adopted)" : " (NOT the code adopted)");
      if (h != WANT_HASH[set][t]) fails++;
      ldpc_free(&c);
    }
  }
  if (print) return 0;
  printf(fails ? "%d FAILED\n" : "every table is as adopted, and ldpc_init builds it\n", fails);
  return fails ? 1 : 0;
}
