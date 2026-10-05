# zstd 1.5.7, as one file

Zstandard 1.5.7 (https://github.com/facebook/zstd, tag v1.5.7), built into one source file by its own
`build/single_file_libs/combine.sh` from a copy of `zstd-in.c` with `ZSTD_MULTITHREAD` left out: the transfer
compresses one chunk at a time, and the WebAssembly build has no threads. `zstd.h` is the library's header as released.
Nothing else is changed. The license is BSD 3-Clause (LICENSE), or GPLv2 (COPYING), at the user's option; the project
takes it under BSD.

The parameters every Lizard build compresses with are pinned in `liblizard/zstd/shim.c`, which is the only code that
calls this library.
