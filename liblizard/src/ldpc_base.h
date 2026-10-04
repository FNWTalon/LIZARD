// The format's LDPC base matrix: rate 3/4 (focus.h FOCUS_RATE) at 8 sub-channels a block, so n_max 5120 and z = 106,
// 12 block rows by 36 data block columns. Each entry is a circulant's shift, -1 where the base graph has no edge; the
// staircase is not in it, being the same for every code (ldpc.c). This IS the format's code. It was written out once
// (2026-09-24) from what ldpc_init generated for it, the 3/4 profile row, seed 1, the xorshift draws and the cycle
// score, so that none of those can move the format's code any more: a change to any of them now moves only the codes
// outside the format. test/ldpc_table.c prints this from the generator and fails if the two differ.
#ifndef OB_LDPC_BASE_H
#define OB_LDPC_BASE_H

enum { LDPC_BASE_Z = 106, LDPC_BASE_RATE = 4, LDPC_BASE_SEED = 1, LDPC_BASE_MB = 12, LDPC_BASE_KB = 36 };

static const short LDPC_BASE[LDPC_BASE_MB][LDPC_BASE_KB] = {
  {   3,  87,  22,  87,  41,  20,  44,  67,  -1,  -1,  -1, 102,  -1,  -1,  -1,   7,  81,  -1,  -1,  -1,  -1,  96,  -1,  -1,  -1,  15,  -1,  -1,  27,  -1,  -1,  -1,   7,  -1,  -1,  -1 },
  {  80,  30,  28,  45,  42,  23,  37,   0,  -1,  -1,  -1,  70,  -1,  13,  -1,  -1,  62,  -1,  -1,  -1,  -1,  -1,  11,  -1,  74,  -1,  -1,  -1,  -1,  -1,  -1,  51,  -1,  -1,  -1,  68 },
  {  58,  72,  29,  92,  57,  83,  53,  51,  -1,  56,  -1,  -1,  -1,  -1,  11,  -1,  -1,  -1,  63,  -1,  -1,  19,  -1,  -1,  -1,  88,  -1,  -1,  -1, 100,  -1,  -1,  -1,  79,  -1,  -1 },
  {  91,  15,  61,  49,  79,  95,  88,  18,  -1,  -1,  -1,  41,  -1,   5,  -1,  -1,  -1,  81,  -1,  -1,  -1,  -1,  -1,  67,  -1,  -1,  -1,  77,  -1,  68,  -1,  -1,  -1,   1,  -1,  -1 },
  {  77,  71,  71,  77,  64,  54,  46, 102,  -1,  -1,  30,  -1,  -1,  -1,  -1,  50,  -1,  -1,  93,  -1,  19,  -1,  -1,  -1,  -1,  -1,  27,  -1,  -1,  -1,  -1,  36,  -1,  -1,  68,  -1 },
  { 100,  93,  44,  43,  11,  51,  63,  36,  -1,  86,  -1,  -1,  -1,  -1,  -1,   9,  -1,  -1,  -1,  55,  67,  -1,  -1,  -1,  -1,  -1,  -1,  47,  20,  -1,  -1,  -1,  -1,  -1, 105,  -1 },
  {  41,  64,  87,  31,  52,  43,  19,   6,  80,  -1,  -1,  -1,  48,  -1,  -1,  -1,  -1,  -1,  -1,  52,  -1,  -1,  -1,  88,  -1,  -1,  -1,  62,  -1,  -1,  52,  -1,  16,  -1,  -1,  -1 },
  {  46,  22,  10,  41,  22,  46,  49,  78,  -1,  63,  -1,  -1,  -1,   6,  -1,  -1,  -1,  28,  -1,  -1,  -1,  98,  -1,  -1,  -1,  -1,  51,  -1,  -1,  -1, 105,  -1,  -1,  -1,  -1,  96 },
  {   9,   7,  18,  25,  72,  79,  60,   8,   2,  -1,  -1,  -1,  -1,  -1,  63,  -1,  -1,  -1,  15,  -1,  -1,  -1,  88,  -1,  39,  -1,  -1,  -1,  -1,  21,  -1,  -1,  46,  -1,  -1,  -1 },
  {  71,  26,  56,  38,  51,  70,   4,  23,  73,  -1,  -1,  -1,  36,  -1,  -1,  -1,  -1,  -1,  -1,  81,  -1,  -1,  -1,  28,  -1,  -1,  31,  -1,  49,  -1,  -1,  -1,  -1,  37,  -1,  -1 },
  {  85,  75,  42,  51,  92,  31,  47,  27,  -1,  -1,  41,  -1,  -1,  -1,  79,  -1,   0,  -1,  -1,  -1,  -1,  -1, 105,  -1,  83,  -1,  -1,  -1,  -1,  -1,  -1,  15,  -1,  -1,  40,  -1 },
  {   9,  21,   4,  48,  43,  77,  67, 105,  -1,  -1, 105,  -1,  66,  -1,  -1,  -1,  -1,  50,  -1,  -1,  90,  -1,  -1,  -1,  -1,  46,  -1,  -1,  -1,  -1,  94,  -1,  -1,  -1,  -1,  50 }
};

#endif
