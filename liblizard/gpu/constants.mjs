// ai: Constants the decoder's stages share with it, in a module that imports nothing (2026-10-01): the classifier and
// ai: proposer plans (plan/classify_gemm.mjs, plan/bank_fcn2.mjs) and the bank's host (bank_fcn2.mjs) take these
// ai: from here, not from decoder.mjs, which imports them; decoder.mjs re-exports them.
export const LEVELS = 5;        // level 0 is the capture; 1 to 4 halve it each time
// The bank's floor on the normalised response, and the deviation added under it so a flat patch does not divide by
// nothing (luma is 0..1).
export const TAU = 0.02, EPS = 0.02;
