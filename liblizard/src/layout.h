// Symbol geometry: LIZARD's border (the thin frame, below) round the picture focus.c owns. Everything that is not
// the picture is static from frame to frame, so a receiver can register on it even in a frame whose picture is torn
// between two display updates.
#ifndef OB_LAYOUT_H
#define OB_LAYOUT_H
#include <stdint.h>
#include "ldpc.h"

// ai: OB_FB bounds a node template's side in modules (acquire.c node_ncc, refine_node); OB_MAX_NODES a side's nodes
enum { OB_FW = 2, OB_FB = 9 * OB_FW, OB_MAX_NODES = 40 };
// CELL_WORD is a flag on the format word's cells (fmt.h). They are painted light or dark like any other cell, so
// every painter reads them through `& 3` and does not care; what the flag says is that a DECODER cannot predict
// them. The word carries a rate the sender chose, so a receiver standing anywhere has the wrong bits for those
// cells, and a registration template that included them would be correlating against noise: measured at 720 with
// the rate mis-stated by one, a symbol that gave 22 blocks of 24 gave 6. So the mesh skips them (acquire.c
// node_ncc, refine_node) and registration does not depend on the word at all, which is what layout.c's
// ob_layout_set_fmt has always claimed.
enum { CELL_DATA = 0, CELL_LIGHT = 1, CELL_DARK = 2, CELL_WORD = 8 };
// Thin frame, from the outside in: one light module, a solid dark line two modules deep, a light
// gap, and one coded band. The solid line is what a receiver finds (four long curves) and what
// gives the displacement across the border; the band is the timing track and gives it along the
// border. The track's cells are 2 x 2 and Manchester coded from a different sequence on each side,
// so no run of track cells is longer than four modules (every 5 x 5 patch of the track sees an edge).
//
// The band carries BOTH the track and the format word (fmt.h), interleaved in runs of OB_BAND_RUN
// cells. The word says which format this is and the sender's display rate; everything else about
// the symbol follows from the format. Counting anchors instead was built and measured on 2026-09-21 and
// cost a quarter of the timing track and four modules of border, for a version read on 93 captures
// of 129 against the word's 99 of 99. fmt.h has the numbers.
//
// CORNER MARKS (OB_THIN_CORNER = S) replace the border's rings inside the square [0, S) at each corner with one solid
// block, because a long dark line is what every page is made of and the line finder cannot tell the border from a bar
// across the page (2026-09-20, straight bars: the thin frame gets 0% where the finder frame it replaced gets 100%). A
// solid block is a 2D template that is rare in the world, and it has ONE feature scale where the finder frame's
// bullseye had five, which is why it can survive the blur that killed that. Depth 3 stays light through the mark (the
// gap carried round it), so the first dark run's inner edge, which is what the line finder fits, stays where it is all
// the way round, and the mark has a cross-section to be found by.
// 12 and gapped because that is where the measurements land (STATUS.md): 8 leaves a core of three modules, too small
// to cluster, and does nothing; 16 fixes less of the straight-bars case than 12 (81% against 100%) and costs 5 to 8% of
// payload everywhere else; 12 costs nothing measurable and is better than no mark at 480, at fill 0.4 and at 45
// degrees. Filled through, a mark shows the line and the core as one run, so it cannot be detected, and it lost half
// the payload at 720 by pulling the side's parabola onto the wrong feature. Edge marks, a centre mark and a 1:1 clock
// track were raced against this border on 2026-09-20 and 21 and lost (STATUS.md); they left the code on 2026-10-25.
// OB_THIN_RESERVE: modules kept clear at each end of a side so the mark never meets a read cell, the mark plus 3 of
// light (2 at least, or blur drags thin_levels' light reference dark): the band's B cells then run exactly beside the
// picture's 2 B modules (focus.h FOCUS_RING).
enum { OB_THIN_CORNER = 12, OB_THIN_RESERVE = OB_THIN_CORNER + 3 };
// How many modules deep every band of the border is that is not the solid line: the light gap, the timing track
// and the format ring, and the light ring inside a mark, which is the gap carried round it.
//
// Two, with OB_CELL 2, so a cell is a 2 x 2 SQUARE and a band is exactly one cell thick. The shape matters for
// the look (runs of equal cells fuse into bars of random length otherwise) and the size is what the measurements force
// in BOTH directions. Smaller fails: 1 x 1 reads the format word on 65 of
// 80 captures against 79, halves the payload under lens distortion (5511 B against 11256 at k1 = 0.06) and starts
// losing frames outright at 480 px and fill 0.4, because binarize calls a pixel dark when it is under its 40 px
// neighbourhood's mean and a one-module cell is the smallest feature the symbol has.
//
// BIGGER ALSO FAILS, and it was measured on 2026-09-21 because a coarser cell looks like it should be blur-proof.
// It is not, and the reason is that the band is not what blur limits: the MESH's sampling accuracy is, and a
// coarser band makes the mesh worse. The two axes were separated, standard twelve cells, everything else held:
//
//   cell    1080    720    480   fill 0.3   defocus 1.5   phone 0.55   format word
//   2 x 2   11256   9556   3576   2990        7621         6156        138 of 138
//   2 x 4   11256   8325   3107   1759        5130         3400        123 of 123
//   4 x 2    6097     --     --     --        3957           --            --
//   4 x 4    7651   3811   1436    674        4485         2902         25 of 118
//
// Longer cells (OB_CELL 4) are pure loss: half as many track cells is half the along-border reference the mesh
// hangs on, and there is no blur gain to show for it. A deeper band (OB_RING_DEEP 4) is free at 1080 and does buy
// blur robustness past about 2.8 px of defocus (2023 B on 16 of 16 frames against 1817 on 15, and it is the only
// setting that returns anything at all at fill 0.4 with 2.5 px of blur), but it costs 30 to 45% at every
// resolution that is actually short of pixels. It also pushes the node lattice off the solid line: OB_NODE_LINE
// is derived from these two, and at depth 4 the five-module template spans gap, gap, gap, gap, band with no dark
// line in it. Sweeping the line back (4.5 / 5.5 / 6.5 / 7.5) recovers some of it and 6.5 is the best of them,
// which is what the 2 x 4 row above was measured at; 7.5 and 4.5 both give ZERO blocks on 16 of 16 registered
// frames, the same "perfect quad, no payload" signature the OB_NODE_LINE comment below records.
//
// The same prediction was made once before in the same units and lost the same way (STATUS.md, "Two-module-deep
// rings were predicted here and measured to be worth nothing ... Not the fix; the white was").
#define OB_RING_DEEP 2
// Modules a cell of those bands runs ALONG the border. With OB_RING_DEEP it sets the cell's shape; the two are
// equal so the cell is square.
#define OB_CELL 2
enum { OB_THIN_LINE = 2, OB_THIN_PITCH = 16,
       OB_THIN_GAP = 1 + OB_THIN_LINE,                  // light, between the line and the track
       OB_THIN_TRACK = OB_THIN_GAP + OB_RING_DEEP,
       OB_THIN_FMT = OB_THIN_TRACK,                     // ONE band: the track and the word share it, see below
       OB_THIN = OB_THIN_TRACK + OB_RING_DEEP };
// Modules of guard interval between the picture and the rest of the border: the picture wrapped round its own edge
// (focus.c paints it, layout.c marks those cells as the picture's), in the border's spare depths past OB_THIN.
#define FOCUS_CP 3
// The border's depth: the corner mark and the guard interval, which cannot share depths (a mark reaching past the
// guard's inner line is a mark inside the picture, focus.c THE RULE). 15.
enum { OB_MARGIN = OB_THIN_CORNER + FOCUS_CP };
// Cell geometry along the border: the middle of cell i, and the first module of the Manchester PAIR that
// module t falls in (a pair is two cells, one dark and one light by construction).
#define OB_CELL_MID (OB_CELL / 2.0f)
// The coded band carries BOTH, interleaved: the timing track and the format word. It used to be two bands four
// modules deep, and depth is what blur has to chew through, where the word has never been the thing that failed.
//
// Interleaved rather than split into halves of a side, because glare lands on a REGION: an interleave loses the
// same fraction of both wherever it falls, a split would take one of them out completely.
//
// How many cells of each in a run before it swaps. 1 alternates every cell, which puts a track cell on BOTH
// sides of every word cell and doubles what blur bleeds into it: measured, the word fell to 15 of 80 captures.
// A longer run leaves most word cells with word neighbours and only the two at each end exposed, while keeping
// what the interleave is for, that glare on a region takes the same fraction of both.
#define OB_BAND_RUN 4
// Modules from `reserve` to the start of track cell j, and of word cell i. A run of each is 2 * OB_BAND_RUN cells.
#define OB_BAND_PAIR(k) (((k) / OB_BAND_RUN) * 2 * OB_BAND_RUN + (k) % OB_BAND_RUN)
#define OB_TRACK_AT(j) (OB_BAND_PAIR(j) * OB_CELL)
#define OB_WORD_AT(i) ((OB_BAND_PAIR(i) + OB_BAND_RUN) * OB_CELL)
// The Manchester pair of TRACK cells nearest module offset `o` from `reserve`: its first track cell index. Track
// cells are 2 * OB_BAND_RUN * OB_CELL modules apart in runs, so this is approximate and only has to land on a
// real pair, which thin_levels is content with: it wants A pair near there, not a particular one.
#define OB_PAIR_J(o) (((o) / (2 * OB_BAND_RUN * OB_CELL)) * OB_BAND_RUN / 2 * 2)
// Where the border's node lattice runs, as a depth. refine_node matches a template five modules across about this
// line, and the template has to REACH THE BAND: the rim, the line and the gap are identical everywhere along an
// edge, so a template made only of them can localize a node across the border but not along it, and the mesh then
// has nothing to hold the picture straight between the corners. Measured the hard way: collapsing the two bands
// into one moved this line from 3.5 to 2.5, which dropped the band out of the template, and an IDEAL image went
// from 12 of 12 blocks to 0 with the finder still scoring a perfect quad.
#define OB_NODE_LINE ((OB_THIN_TRACK + OB_RING_DEEP) / 2.0f)
// The middle of each band, as a depth to sample at.
#define OB_GAP_MID (OB_THIN_GAP + OB_RING_DEEP / 2.0f)
#define OB_TRACK_MID (OB_THIN_TRACK + OB_RING_DEEP / 2.0f)
#define OB_FMT_MID (OB_THIN_FMT + OB_RING_DEEP / 2.0f)
int ob_thin_track(int side, int j);          // 1 where TRACK CELL j of side 0..3 (top, right, bottom, left) is dark
int ob_thin_cells(int side, int reserve);    // cells of the band on a side of `side` modules, both kinds
int ob_track_cells(int side, int reserve);   // of those, the timing track's

// ai: The border round a picture: a square of w = h modules (2 B + 30 in ring B), OB_MARGIN deep, its corner marks
// ai: OB_THIN_CORNER and the band reserve OB_THIN_RESERVE, which every reader takes from here.
typedef struct {
  int w, h;
  int corner, reserve, margin;
  int nx, ny;                     // mesh nodes per axis, finders included
  float node_x[OB_MAX_NODES], node_y[OB_MAX_NODES];   // module coordinates of node centres
  uint8_t node_mark[OB_MAX_NODES * OB_MAX_NODES];     // 1 where a lattice point carries a mark
  uint8_t *kind;                  // w * h, CELL_*
} ob_layout_t;

// The border for a symbol `side` modules square (64 at least). 0, or -1.
int ob_layout_init(ob_layout_t *L, int side);
void ob_layout_free(ob_layout_t *L);
// Repaint the thin frame's format word from an encoded codeword (fmt.h). Nothing else in the frame
// moves, so a sender may change it between the frames of a running transfer.
void ob_layout_set_fmt(ob_layout_t *L, const uint8_t *cw);

uint32_t ob_crc32(const uint8_t *p, int n);

#endif
