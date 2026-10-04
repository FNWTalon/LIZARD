// The format's LDPC base matrix (src/ldpc_base.h) against the generator it was written out from (src/ldpc.c
// ldpc_generate). Prints the table as the header holds it, then fails if the generator no longer makes it, or if
// ldpc_init's code is not the one the table describes.
//   gcc -O2 -Wall -Isrc -o build/ldpc_table test/ldpc_table.c src/ldpc.c && build/ldpc_table
// With `print` it prints the table and checks nothing: what a deliberate change to the format's code would paste in.
#include "ldpc.h"
#include "ldpc_base.h"
#include <stdio.h>
#include <string.h>

// FNV-1a over H's rows and columns: 2ca2bc24 for the format's code as ldpc_init built it on 2026-09-24, before the table.
static unsigned h_hash(const ldpc_t *c) {
  unsigned h = 2166136261u;
  for (int i = 0; i <= c->m; i++) h = (h ^ (unsigned)c->row_ptr[i]) * 16777619u;
  for (int e = 0; e < c->edges; e++) h = (h ^ (unsigned)c->col_idx[e]) * 16777619u;
  return h;
}

int main(int argc, char **argv) {
  static int gen[LDPC_NB][LDPC_NB];
  if (ldpc_generate(LDPC_BASE_Z, LDPC_BASE_RATE, LDPC_BASE_SEED, gen) != LDPC_BASE_MB) { printf("FAIL: the generator's rate %d has not %d block rows\n", LDPC_BASE_RATE, LDPC_BASE_MB); return 1; }
  printf("static const short LDPC_BASE[LDPC_BASE_MB][LDPC_BASE_KB] = {\n");
  for (int r = 0; r < LDPC_BASE_MB; r++) {
    printf("  {");
    for (int j = 0; j < LDPC_BASE_KB; j++) printf(" %3d%s", gen[r][j], j + 1 < LDPC_BASE_KB ? "," : "");
    printf(" }%s\n", r + 1 < LDPC_BASE_MB ? "," : "");
  }
  printf("};\n");
  if (argc > 1 && !strcmp(argv[1], "print")) return 0;

  int fails = 0;
  for (int r = 0; r < LDPC_BASE_MB; r++) for (int j = 0; j < LDPC_NB; j++) {
    const int want = j < LDPC_BASE_KB ? LDPC_BASE[r][j] : -1;
    if (gen[r][j] != want) { if (fails++ < 8) printf("FAIL: row %d column %d: the generator says %d, the table %d\n", r, j, gen[r][j], want); }
  }
  // The code ldpc_init builds for the format (8 sub-channels a block, two slots a coefficient: n_max 5120) has to be
  // the table's, read back from its block-row view.
  ldpc_t c;
  if (ldpc_init(&c, 8 * 320 * 2, LDPC_BASE_RATE, LDPC_BASE_SEED) || c.z != LDPC_BASE_Z || c.mb != LDPC_BASE_MB) { printf("FAIL: ldpc_init did not build the format's code\n"); return 1; }
  for (int r = 0; r < c.mb; r++) {
    int seen = 0;
    for (int s = c.lay_ptr[r]; s < c.lay_ptr[r + 1]; s++, seen++)
      if (LDPC_BASE[r][c.lay_col[s]] != c.lay_shift[s]) { fails++; printf("FAIL: ldpc_init row %d column %d shift %d, the table %d\n", r, c.lay_col[s], c.lay_shift[s], LDPC_BASE[r][c.lay_col[s]]); }
    int want = 0;
    for (int j = 0; j < LDPC_BASE_KB; j++) want += LDPC_BASE[r][j] >= 0;
    if (seen != want) { fails++; printf("FAIL: ldpc_init row %d has %d circulants, the table %d\n", r, seen, want); }
  }
  const unsigned h = h_hash(&c);
  printf("format code: n %d k %d z %d, %d block rows, %d edges, H %08x%s\n", c.n, c.k, c.z, c.mb, c.edges, h, h == 0x2ca2bc24u ? " (as generated before the table)" : " (NOT the code generated before the table)");
  if (h != 0x2ca2bc24u) fails++;
  ldpc_free(&c);
  printf(fails ? "%d FAILED\n" : "the table is the generator's, and ldpc_init builds it\n", fails);
  return fails ? 1 : 0;
}
