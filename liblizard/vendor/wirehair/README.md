# Wirehair, vendored

| | |
|---|---|
| Upstream | https://github.com/catid/wirehair |
| Revision | `067ca7cdb66aed424ec23f97557429bf791c6f0c` (2026-07-24) |
| License | BSD 3-Clause (`LICENSE`) |
| Copied | the 15 library sources, all headers, the `.inc` tables, `include/wirehair/`, `LICENSE` |
| Not copied | `bench/`, `test/`, `python/`, `experiments/`, `tables/`, `cmake/`, `codec/*Test.cpp`, `codec/fuzz/`, `codec/WirehairV2Bench.cpp` |
| Patches | None: every file is byte-identical to upstream. |

The outer fountain code of a transfer: a receiver rebuilds a chunk from any N + 0.02 of its blocks on average, whichever
frames it missed. Only the V2 API is used (`../../wirehair/shim.cpp`), whose 32-byte wire profile pins the encoding.
`-DWIREHAIR_BUILDING=1` is passed by the builds (`../../build.sh wirehair`, `../../core/CMakeLists.txt`) rather than
patched in.
