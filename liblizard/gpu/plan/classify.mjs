// ai: The installed classifier and its int8 twins (the host's INSTALLED; ../../gen/gen.mjs builds both).

// ai: The installed classifier (gpu/decoder.mjs INSTALLED): cnn-v3, run on every kept peak. A path under liblizard/
// ai: since 2026-10-01 (a name under gpu/cnn/ before): the installed app names every file by a hash of its bytes and
// ai: rewrites only whole literal paths (lizard-web/pwa/hash.mjs), and the harness page takes either.
export const CLASSIFIER = "gpu/cnn/weights-v3.safetensors";
// ai: The installed classifiers' int8 twins, paths under liblizard/ (names under gpu/cnn/ until 2026-10-01, which the built
// ai: app's renamed files no longer matched: it would have run the float classifiers without a word): a float file's
// ai: q8 form (cnn/q8c.mjs), run where the decoder runs at precision int8 and its measurement on the first stream
// ai: batch's frames says int8 is faster there (gpu/decoder.mjs create's twins, measureTwins). Callers pass the float files; the
// ai: decoder finds the twin by the file the document was loaded from (cnn/netfile.mjs loadNet's source). Each twin's
// ai: CQ81 reference is build/cnn/reference-<X>.bin for weights-<X>.safetensors, as the float files' (cnn/check.mjs).
export const INT8_TWINS = { "gpu/cnn/weights.safetensors": "gpu/cnn/weights-q8.safetensors", "gpu/cnn/weights-v3.safetensors": "gpu/cnn/weights-v3-q8.safetensors" };
