#!/bin/sh
# ./build.sh        -> build/ob.mjs + ob.wasm (needs emsdk: source <emsdk>/emsdk_env.sh)
# NOSIMD=-DOB_SCALAR OUT=build/nosimd ./build.sh -> the same source without the vector paths, for a lab check that is
#   not published (scripts/exp/simd_check.mjs)
# ./build.sh native -> the C compiled natively under ASan and UBSan, warnings on, nothing kept (a compile check)
# ./build.sh verify -> build/verify and build/verify-scalar, for a lab check that is not published (scripts/exp/points_check.mjs)
# ./build.sh wirehair -> build/wirehair.mjs, the outer fountain for end-to-end runs
# ./build.sh zstd     -> build/zstd.mjs, the transfer's compression (vendor/zstd through zstd/shim.c)
# ./build.sh app -> ../lizard-web/app/, the installed web app (../lizard-web/pwa/build.mjs: content-addressed, its service worker; node only)
# -ffp-contract=off: the reference is exact (2026-09-24), and a compiler that fuses a multiply and an add into
# one FMA rounds once where the source rounds twice, so an FMA machine's build would paint other bytes.
set -e
cd "$(dirname "$0")"
SRC="src/ldpc.c src/layout.c src/enc.c src/acquire.c src/demap.c src/dec.c src/fft.c src/rs.c src/fmt.c src/focus.c src/any.c src/shake.c src/xfer.c src/wasm.c vendor/blake3/blake3.c vendor/blake3/blake3_dispatch.c vendor/blake3/blake3_portable.c"
# BLAKE3 for the transfer's hash (src/xfer.h): the official C, portable only, which is all wasm can run of it, so the
# native check compiles the same paths. No threads.
B3="-Ivendor/blake3 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512 -DBLAKE3_USE_NEON=0"
if [ "$1" = app ]; then
  # The installed app (2026-10-01, a content-hashed PWA): build/ob.* and build/wirehair.mjs
  # as they are, so build those first when the C or the shim changed. An optional second argument names another out dir
  # (relative to liblizard/).
  OUTDIR="${2:-../lizard-web/app}"
  case "$OUTDIR" in /*) ;; *) OUTDIR="$PWD/$OUTDIR" ;; esac
  exec node ../lizard-web/pwa/build.mjs "$OUTDIR"
fi
if [ "$1" = wirehair ]; then
  # The project's own shim (wirehair/shim.cpp), compiled by path beside the vendored source.
  WH=vendor/wirehair
  mkdir -p build
  em++ -std=c++17 -Oz -DWIREHAIR_BUILDING=1 -I $WH/include -I $WH \
    $WH/wirehair.cpp $WH/gf256.cpp $WH/WirehairCodec.cpp $WH/WirehairTools.cpp $WH/codec/WirehairV2Codec.cpp \
    $WH/codec/WirehairV2GF16.cpp $WH/codec/WirehairV2Peel.cpp $WH/codec/WirehairV2Plan.cpp $WH/codec/WirehairV2Policy.cpp \
    $WH/codec/WirehairV2Precode.cpp $WH/codec/WirehairV2PrecodeDecode.cpp $WH/codec/WirehairV2PrecodeEncode.cpp \
    $WH/codec/WirehairV2Profile.cpp $WH/codec/WirehairV2Seeds.cpp $WH/codec/WirehairV2Solve.cpp wirehair/shim.cpp \
    -o build/wirehair.mjs -sMODULARIZE=1 -sEXPORT_ES6=1 -sSINGLE_FILE=1 -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=node,web,worker \
    -sEXPORTED_FUNCTIONS=_lizard_wh_init,_lizard_wh_encoder_create,_lizard_wh_decoder_create,_lizard_wh_encode,_lizard_wh_decode,_lizard_wh_recover,_lizard_wh_free,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=HEAPU8,getValue,setValue
  ls -l build/wirehair.mjs
  exit 0
fi
if [ "$1" = zstd ]; then
  # zstd as one file (vendor/zstd/README.md) under the project's shim (zstd/shim.c), which pins the parameters.
  mkdir -p build
  emcc -Oz -DNDEBUG -I vendor/zstd vendor/zstd/zstd.c zstd/shim.c -o build/zstd.mjs \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sSINGLE_FILE=1 -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=node,web,worker \
    -sEXPORTED_FUNCTIONS=_lizard_zstd_bound,_lizard_zstd_compress,_lizard_zstd_decompress,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=HEAPU8
  ls -l build/zstd.mjs
  exit 0
fi
if [ "$1" = verify ]; then
  # The thin frame's point scans next to the plain scan they replaced, which aborts on any difference: scripts/exp/points_check.mjs.
  # ./build.sh, not "$0": the cd above made a relative $0 (liblizard/build.sh from the root) point nowhere.
  NOSIMD="-msimd128 -DOB_VERIFY_POINTS" OUT=build/verify ./build.sh
  NOSIMD="-DOB_SCALAR -DOB_VERIFY_POINTS" OUT=build/verify-scalar ./build.sh
  exit 0
fi
if [ "$1" = native ]; then
  gcc -O2 -g -ffp-contract=off -Wall -Wextra -fsanitize=address,undefined $B3 -c $SRC && rm -f ./*.o
  exit 0
fi
mkdir -p ${OUT:-build}
emcc -O3 ${NOSIMD:--msimd128} -ffp-contract=off -DNDEBUG $B3 $SRC -o ${OUT:-build}/ob.mjs \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node,web,worker -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=8388608 \
  -sEXPORTED_FUNCTIONS=_ob_setup,_ob_tiles,_ob_code_n,_ob_code_k,_ob_set_opts,_ob_pilots,_focus_setup,_focus_setup_tiers,_focus_any_setup,_focus_any_rx,_focus_any_held,_focus_any_max_blocks,_focus_any_which,_focus_any_n,_focus_any_blocks,_focus_any_fmt,_focus_any_builds,_focus_bitmap_set,_focus_bitmap_get,_focus_subch,_focus_blocks,_focus_side,_focus_cell,_focus_quiet,_focus_fmt_rx_ptr,_focus_fmt_fps_set,_focus_parity_set,_focus_pilot_get,_focus_any_pilot_get,_focus_tx,_focus_tx_rgba,_focus_rs_geom,_focus_rx,_focus_dbg,_focus_dbg_sym,_focus_dbg_coef,_focus_blk_its,_focus_blk_est,_ob_heap_top,_ob_fft_bench,_fft_set_radix,_fft_get_radix,_ob_tx,_ob_rx,_ob_last,_ob_result_size,_ob_prof,_ob_prof_reset,_ob_rx_rgba,_ob_luma_of,_ob_luma_strip,_ob_test_binarize,_ob_test_sample,_ob_test_marks,_ob_test_mark_scan_out,_ob_test_mark_cfg_out,_ob_test_mark_quads_out,_ob_test_line_points,_ob_test_mesh_tables_out,_ob_test_corner_coords_out,_ob_test_homography,_ob_test_fmt_plan,_ob_test_fmt_cells,_ob_test_thin_plan,_ob_test_thin_score_out,_ob_test_finder_out,_focus_rx_acquire,_focus_rx_acquire_quad,_focus_rx_quad,_focus_rx_finish,_focus_shape,_focus_pos_ptr,_focus_tables_out,_focus_spectrum_out,_focus_rx_ldpc,_focus_block_subs,_focus_ldpc_tables,_focus_block_tiers,_focus_rx_bits,_focus_rx_assemble,_focus_fmt_check_out,_focus_grid_out,_ob_test_crc_pack,_ob_test_crc_hash,_ob_test_crc_gate_out,_ob_dbg_hash,_ob_dbg_llr,_ob_kind,_stream_fill,_xfer_layout,_xfer_id_of,_xfer_kind_of,_xfer_hdr_write,_xfer_hdr_parse,_xfer_mf_parse,_xfer_chunk_ok,_xfer_manifest_blocks,_xfer_manifest_write,_xfer_manifest_check,_xfer_root,_xfer_chunk_cv,_xfer_b3_hash,_xfer_b3_sub,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP32
ls -l ${OUT:-build}/ob.wasm
