// ai: Constants the decoder's stages share with it, in a module that imports nothing (2026-10-01): the installed app names
// ai: every module by a hash of its bytes (lizard-web/pwa/hash.mjs), so no two modules may name each other, and the classifier
// ai: and proposer kernels (wgsl/classify_gemm.mjs, wgsl/bank_fcn2.mjs) and the banks' hosts (bank_fcn2.mjs, bank_fcn.mjs)
// ai: took these from decoder.mjs, which imports them. decoder.mjs re-exports them all, so every other importer is unchanged.
export const LEVELS = 5;        // level 0 is the capture; 1 to 4 halve it each time
// The bank's floor on the normalised response, and the deviation added under it so a flat patch does not divide by
// nothing (luma is 0..1).
export const TAU = 0.02, EPS = 0.02;
// ai: The WGSL language feature the int8 nets need (dot4I8Packed, pack4xI8): a feature of the browser's WGSL, not of
// ai: the adapter, so no device asks for it.
export const INT8 = "packed_4x8_integer_dot_product";
