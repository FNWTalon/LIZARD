#!/bin/bash
# ai: arm64's own kernels on a phone: ldpc_neon.c (the LDPC against the scalar loop and the 128-bit kernel), fft_dec.c
# ai: (the decoder's transform against the definition), crc_check.c (the CRC instructions against the bit loop),
# ai: mesh_neon.c (the registration mesh, built with its NEON forms and with the scalar twins: the same hash), built
# ai: with the NDK, pushed to /data/local/tmp/lizard_kernels/, run pinned to cpu7 (a prime core on the S26) under the
# ai: phone lock, the screen woken first (a dozing phone parks its cores). Exit 1 on any failure.
# ai:   liblizard/test/neon_kernels.sh   (from anywhere; about 1 min; NDK, LIZ_PHONE_LOCK, CORE as in neon_check.sh)
set -e
here=$(cd "$(dirname "$0")" && pwd)
NDK=${NDK:-${ANDROID_NDK_HOME:-${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.1.12297006}}
CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
out=$here/../build/neon_kernels; P=/data/local/tmp/lizard_kernels; CORE=${CORE:-80}
mkdir -p "$out"
F="-O3 -ffp-contract=off -DNDEBUG -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -I$here/../src"
$CC $F -o "$out/ldpc_neon" "$here/ldpc_neon.c" -lm
$CC $F -o "$out/fft_dec" "$here/fft_dec.c" -lm
$CC $F -o "$out/crc_check" "$here/crc_check.c" "$here/../src/layout.c" "$here/../src/fmt.c" "$here/../src/rs.c" -lm
S=$here/../src; codec="$S/layout.c $S/fmt.c $S/rs.c $S/ldpc.c $S/fft.c $S/lizard_phy.c $S/any.c $S/shake.c"
$CC $F -o "$out/mesh_neon" "$here/mesh_neon.c" $codec -lm
$CC $F -DOB_SCALAR -o "$out/mesh_scalar" "$here/mesh_neon.c" $codec -lm
exec 9>"${LIZ_PHONE_LOCK:-/tmp/lizard-phone.lock}"; flock 9
adb shell mkdir -p $P >/dev/null
adb push "$out/ldpc_neon" "$out/fft_dec" "$out/crc_check" "$out/mesh_neon" "$out/mesh_scalar" $P/ >/dev/null
fail=0
for t in "ldpc_neon 20000 500 5" "fft_dec 30" "crc_check"; do
  echo "== $t"
  adb shell "input keyevent KEYCODE_WAKEUP; taskset $CORE $P/$t; echo exit \$?" | tee /dev/stderr | tail -1 | grep -q 'exit 0' || fail=1
done 2>&1
echo "== mesh_neon, mesh_scalar"
hv=$(adb shell "input keyevent KEYCODE_WAKEUP; taskset $CORE $P/mesh_neon" | tee /dev/stderr | grep 'mesh hash')
hs=$(adb shell "taskset $CORE $P/mesh_scalar" | tee /dev/stderr | grep 'mesh hash')
[ -n "$hv" ] && [ "$hv" = "$hs" ] || { echo "the mesh's NEON forms and scalar twins differ"; fail=1; }
exit $fail
