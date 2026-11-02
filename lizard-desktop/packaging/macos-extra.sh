#!/bin/sh
# ai: The files the macOS package carries beside liblizard_desktop.dylib (build.gradle.kts bundle, from
# ai: build/native/extra/): Homebrew's libusb and Aravis 0.8 with every library they load that is not the system's,
# ai: each one's install name and references rewritten to @loader_path (the folder they are extracted to together),
# ai: each signed again ad hoc (arm64 refuses a library whose signature the rewrite broke); in licenses/ each
# ai: formula's license files. Run after `brew install libusb aravis` (the desktop workflow's macOS job).
#   packaging/macos-extra.sh <out folder>
set -eu
OUT=$1
mkdir -p "$OUT/licenses"
OUT=$(cd "$OUT" && pwd)
BREW=$(brew --prefix)

# ai: a library's references outside the system (/usr/lib, /System), as written in it (absolute, or @rpath/@loader_path)
refs() { otool -L "$1" | tail -n +2 | awk '{print $1}' | grep -v -E '^(/usr/lib/|/System/)' || true; }
resolve() {   # ref, the library referring to it
  case $1 in
    @loader_path/*) echo "$(dirname "$2")/${1#@loader_path/}" ;;
    @rpath/*) for d in "$BREW/lib" "$(dirname "$2")"; do [ -f "$d/${1#@rpath/}" ] && { echo "$d/${1#@rpath/}"; return; }; done; echo "$1" ;;
    *) echo "$1" ;;
  esac
}

# ai: the closure: each library's references followed, each file once (by its real path)
todo="$BREW/opt/libusb/lib/libusb-1.0.0.dylib $BREW/opt/aravis/lib/libaravis-0.8.0.dylib"
done_list=""
while [ -n "$todo" ]; do
  set -- $todo
  cur=$1
  shift
  todo="$*"
  [ -e "$cur" ] || { echo "macos-extra: $cur not found" >&2; exit 1; }
  lib=$(realpath "$cur")
  case " $done_list " in *" $lib "*) continue ;; esac
  done_list="$done_list $lib"
  for r in $(refs "$lib"); do todo="$todo $(resolve "$r" "$lib")"; done
done
# ai: copied under the name each is referred to by (its install name's leaf: libglib-2.0.0.dylib, not the versioned file)
for lib in $done_list; do
  leaf=$(basename "$(otool -D "$lib" | tail -n 1)")
  cp "$lib" "$OUT/$leaf"
  chmod u+w "$OUT/$leaf"
done
for f in "$OUT"/*.dylib; do
  install_name_tool -id "@loader_path/$(basename "$f")" "$f"
  for r in $(refs "$f"); do
    leaf=$(basename "$r")
    [ -f "$OUT/$leaf" ] || { echo "macos-extra: $(basename "$f") refers to $r, which is not carried" >&2; exit 1; }
    install_name_tool -change "$r" "@loader_path/$leaf" "$f"
  done
  codesign --force --sign - "$f"
done
# ai: every reference now inside the folder or the system's
for f in "$OUT"/*.dylib; do
  refs "$f" | grep -v '^@loader_path/' && { echo "macos-extra: $(basename "$f") still refers outside" >&2; exit 1; } || true
done
# ai: the license files of each formula the libraries came from (its keg's COPYING, LICENSE and the like)
for lib in $done_list; do
  keg=$(echo "$lib" | sed -n 's|.*/Cellar/\([^/]*\)/\([^/]*\)/.*|\1/\2|p')
  [ -n "$keg" ] || continue
  f=${keg%/*}
  [ -f "$OUT/licenses/$f.txt" ] && continue
  {
    for l in "$BREW/Cellar/$keg"/COPYING* "$BREW/Cellar/$keg"/LICENSE* "$BREW/Cellar/$keg"/LICENCE*; do
      [ -f "$l" ] && { printf '%s %s (Homebrew): %s\n\n' "$f" "${keg#*/}" "$(basename "$l")"; cat "$l"; printf '\n\n'; }
    done
  } > "$OUT/licenses/$f.txt"
  [ -s "$OUT/licenses/$f.txt" ] || printf '%s %s (Homebrew): %s\n' "$f" "${keg#*/}" "$(brew info --json=v2 "$f" | python3 -c 'import json,sys; print(json.load(sys.stdin)["formulae"][0].get("license") or "license not stated")')" > "$OUT/licenses/$f.txt"
done
echo "macos-extra: $(ls "$OUT"/*.dylib | wc -l | tr -d ' ') libraries, $(ls "$OUT/licenses" | wc -l | tr -d ' ') licenses in $OUT"
