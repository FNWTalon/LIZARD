// ai: Test-only LDPC decoders in floating point, for measuring what src/ldpc.c's integer min-sum leaves behind:
// ai: layered sum-product (belief propagation) and layered min-sum with a scale and an offset. They run on the
// ai: format's ldpc_t as ldpc_init builds it, over the same schedule as ldpc_decode (a layer is one block row, the
// ai: sweep direction alternating by iteration), and take LLRs in natural units, codeword order, > 0 meaning bit 0.
// ai: Nothing in src/ uses them.
#ifndef OB_LDPC_REF_H
#define OB_LDPC_REF_H
#include "../src/ldpc.h"

enum { REF_BP, REF_MINSUM };

typedef struct {
  int rule;       // ai: REF_BP (sum-product) or REF_MINSUM
  int max_iter;
  int stop;       // ai: 1 stops at the first clear syndrome; 0 runs every iteration and judges the last
  double alpha;   // ai: REF_MINSUM: the scale on a check's minimum (1 for offset min-sum)
  double beta;    // ai: REF_MINSUM: the offset taken off it after scaling, in LLR units (0 for normalised min-sum)
} ref_opts_t;

typedef struct {
  const ldpc_t *c;
  int deg_max;            // ai: the most edges a check has
  double *L;              // ai: posteriors, codeword order
  double *R;              // ai: check-to-bit messages, one an edge, in c->col_idx's order
  double *q, *t, *bw;     // ai: one check's bit-to-check messages; BP's tanh of each and their suffix products
} ref_dec_t;

// ai: Workspace for one decoder on code c, which must outlive it. Returns 0 on success.
int ref_init(ref_dec_t *d, const ldpc_t *c);
void ref_free(ref_dec_t *d);

// ai: Decodes llr (n values) into out (n hard decisions, codeword order). Returns the iterations to a clear syndrome
// ai: (stop = 1), max_iter if the last iteration's is clear (stop = 0), or -1.
int ref_decode(ref_dec_t *d, const float *llr, uint8_t *out, const ref_opts_t *o);

// ai: focus.c's stall rule (FOCUS_STALL_IT, FOCUS_STALL_RATIO), restated here because focus.c keeps it private: after
// ai: iteration 9, give up if the violated checks are above 0.95 of their count after iteration 1.
#define REF_STALL_IT 9
#define REF_STALL_RATIO 0.95f

// ai: The soft stage's quantiser (focus.c): an LLR clipped to +-10, times 8, rounded to nearest even. [-80, 80].
int8_t ref_quant8(float llr);

// ai: An arm: one decoder with its settings, as test/ldpc_awgn.c and test/ldpc_replay.c compare them.
typedef struct {
  const char *name;
  int today;       // ai: 1: src/ldpc.c's decoder on the int8 soft values; 0: ref_decode
  int norm;        // ai: today: its min-sum scale in sixteenths (13 is the format's)
  int stall;       // ai: today: 1 applies the stall rule above, as focus.c does
  int quant;       // ai: ref: 1 decodes the int8 soft values over 8, 0 the unquantised LLRs
  ref_opts_t ref;  // ai: ref's rule; ref.max_iter is the cap for both kinds
} ref_arm_t;

// ai: Runs arm a on one block. q8: the int8 soft values; fl: the unquantised LLRs (read only by a ref arm with quant
// ai: 0); buf: n floats of scratch; code: the arm's code, whose workspace and norm a today arm uses; ref: a decoder on
// ai: it. Returns what the decoder returned (iterations to a clear syndrome, -1 at the cap, -3 given up by the stall
// ai: rule) and puts the iterations it ran in *spent.
int ref_arm_decode(const ref_arm_t *a, ldpc_t *code, ref_dec_t *ref, const int8_t *q8, const float *fl, float *buf,
                   uint8_t *out, int *spent);

// ai: The BI-AWGN channel the tools share. A generator is seeded from (stream, index), so a frame's data and unit
// ai: noise depend on those alone: every arm and every noise level sees the same frames, scaled.
typedef struct { uint64_t s; } ref_rng_t;
ref_rng_t ref_rng_at(uint64_t stream, uint64_t index);
double ref_uni(ref_rng_t *r);     // ai: [0, 1)
double ref_gauss(ref_rng_t *r);   // ai: unit normal
double ref_mi_awgn(double sigma); // ai: the channel's information a bit at noise sigma, BPSK +-1
double ref_sigma_for(double mi);  // ai: its inverse

#endif
