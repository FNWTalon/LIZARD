#include "ob.h"
#include <stdlib.h>
#include <string.h>

void ob_encode(const ob_layout_t *L, const uint8_t *blocks, uint8_t *modules) {
  const ldpc_t *c = &L->code;
  int cells = L->w * L->h, B = L->block_bytes;
  for (int i = 0; i < cells; i++) modules[i] = (L->kind[i] & 3) == CELL_DARK;
  uint8_t *data = calloc((size_t)c->k, 1), *cw = malloc((size_t)c->n), *buf = malloc((size_t)B + 4);
  for (int t = 0; t < L->tiles; t++) {
    memcpy(buf, blocks + (size_t)t * B, (size_t)B);
    uint32_t crc = ob_crc32(buf, B);
    for (int i = 0; i < 4; i++) buf[B + i] = (uint8_t)(crc >> (8 * i));
    memset(data, 0, (size_t)c->k);
    for (int i = 0; i < (B + 4) * 8; i++) data[i] = (buf[i >> 3] >> (7 - (i & 7))) & 1;
    ldpc_encode(c, data, cw);
    for (int b = 0; b < c->n; b++) { int cell = L->tile_cells[t][b]; modules[cell] = cw[b] ^ ob_scramble(cell); }
    for (int i = 0; i < L->tile_npad[t]; i++) { int cell = L->tile_pad[t][i]; modules[cell] = (uint8_t)ob_scramble(cell); }
  }
  free(data); free(cw); free(buf);
}
