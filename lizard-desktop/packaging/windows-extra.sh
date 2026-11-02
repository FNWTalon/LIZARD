#!/bin/sh
# ai: The files the Windows package carries beside lizard_desktop.dll (build.gradle.kts bundle, from build/native/extra/),
# ai: made on Linux with MinGW-w64 (the desktop workflow's tree job), every download checked by its sha256:
# ai: - Aravis 0.8 and the DLLs it loads, MSYS2's ucrt64 builds: the GenICam cameras (liblizard/core/cam/aravis.cpp loads
# ai:   libaravis-0.8-0.dll from the native folder), and libusb-1.0.dll, which the phone on the cable shares
# ai:   (liblizard/core/cam/aoa.cpp); their closure found by their import tables (the rest is Windows' own, UCRT inbox
# ai:   since Windows 10): 12 DLLs, about 8 MB;
# ai: - lizard_usb.exe (native/usb/lizard_usb.c), built against libwdi 1.5.1 (LGPL-3.0, linked statically) with the
# ai:   WinUSB co-installers of Microsoft's WDF 1.11 redistributable, as libwdi's own build takes them;
# ai: - licenses/: each one's license text, which build.gradle.kts appends to the package's.
#   packaging/windows-extra.sh <out folder> [<work folder>]
set -eu
OUT=$1
WORK=${2:-$(mktemp -d)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/licenses" "$WORK"
OUT=$(cd "$OUT" && pwd)
WORK=$(cd "$WORK" && pwd)

fetch() {   # url file sha256
  [ -f "$WORK/$2" ] && echo "$3  $WORK/$2" | sha256sum -c --status && return 0
  curl -sfL --retry 3 -o "$WORK/$2" "$1" || { echo "windows-extra: $1 not fetched" >&2; exit 1; }
  echo "$3  $WORK/$2" | sha256sum -c --status || { echo "windows-extra: $2's sha256 is not $3" >&2; exit 1; }
}

# ai: MSYS2's packages (repo.msys2.org/mingw/ucrt64), pinned; to move one, take the new file name and SHA256SUM from
# ai: the repository's ucrt64.db and check the closure again (x86_64-w64-mingw32-objdump -p, DLL Name)
M=https://repo.msys2.org/mingw/ucrt64
P=mingw-w64-ucrt-x86_64
while read -r name sum; do
  fetch "$M/$P-$name-any.pkg.tar.zst" "$P-$name.pkg.tar.zst" "$sum"
  mkdir -p "$WORK/msys2"
  zstd -dcq "$WORK/$P-$name.pkg.tar.zst" | tar -x -C "$WORK/msys2"
done <<'EOF'
aravis-0.8.33-4 273e1346b2b748fb38dfd0bb975d95bf4d11b197970f7bd0ebd41c71182c7d12
glib2-2.90.1-1 9a7826349279bcca0c935b01880d812be9de9609aef2205be68fd7a6128c1695
libxml2-2.15.4-3 e774c85139c004ce9b4085474d6b3450526c718d2e8c555a87e47adc11f376b1
zlib-1.3.2-3 130e6ef04c654fc2150a177600aa5ed0035087665a893479d2c85ba7485d5704
libusb-1.0.30-1 20106e6ff31b4581bb111b28a68b6b79a50dc3e5fc3033944e1c4b3ff8cd812e
gettext-runtime-1.0-1 ba693dda4ac375af76ce481ff3a6e7481286546cc7dc6d56c7021dae34084157
libiconv-1.19-1 9a500f38c2b91808741c62fae746b3e9110b33a1ecf5c30fa0c66dbedddf7e16
pcre2-10.49-1 63172179db724ac71e2439a7981134c1bcab8049bdddfd349b32b8f76ed52d1e
libffi-3.8.0-1 99ad12f4ecfa00a889ef9c5c0188592368a5bac109d0c508ce4e04f29ea95a76
EOF
B=$WORK/msys2/ucrt64/bin
for d in libaravis-0.8-0 libglib-2.0-0 libgobject-2.0-0 libgio-2.0-0 libgmodule-2.0-0 libxml2-16 zlib1 libusb-1.0 libintl-8 \
         libiconv-2 libpcre2-8-0 libffi-8; do
  cp "$B/$d.dll" "$OUT/"
done
# ai: the closure checked: every DLL these import is one of them or Windows' own
for f in "$OUT"/*.dll; do
  x86_64-w64-mingw32-objdump -p "$f" | sed -n 's/^\s*DLL Name: //p'
done | sort -u | while read -r dep; do
  [ -f "$OUT/$dep" ] && continue
  case $(echo "$dep" | tr 'A-Z' 'a-z') in
    api-ms-win-*|kernel32.dll|advapi32.dll|user32.dll|shell32.dll|shlwapi.dll|ole32.dll|ws2_32.dll|iphlpapi.dll|dnsapi.dll|bcrypt.dll) ;;
    *) echo "windows-extra: $dep, which the DLLs import, is not carried" >&2; exit 1 ;;
  esac
done
L=$WORK/msys2/ucrt64/share/licenses
for p in glib2 libxml2 zlib libusb gettext-runtime libiconv pcre2 libffi; do
  for f in "$L/$p"/*; do [ -f "$f" ] && { printf '%s (MSYS2 %s): %s\n\n' "$p" ucrt64 "$(basename "$f")"; cat "$f"; printf '\n\n'; }; done > "$OUT/licenses/$p.txt"
done
fetch https://raw.githubusercontent.com/AravisProject/aravis/0.8.33/COPYING aravis-COPYING dc626520dcd53a22f727af3ee42c770e56c97a64fe3adb063799d8ab032fe551
{ printf 'Aravis 0.8.33 (MSYS2 ucrt64): COPYING\n\n'; cat "$WORK/aravis-COPYING"; } > "$OUT/licenses/aravis.txt"

# ai: libwdi 1.5.1 and the WDF 1.11 redistributable (wdfcoinstaller.msi, Microsoft's LinkID 253170): its x64
# ai: WdfCoInstaller01011.dll and winusbcoinstaller2.dll laid out as libwdi's configure looks for them under a WDK
fetch https://github.com/pbatard/libwdi/archive/refs/tags/v1.5.1.tar.gz libwdi-1.5.1.tar.gz a695e93db0977dfdc5c6a99a4ea91b22f9027547d0177b2a0f3075078643c929
fetch https://download.microsoft.com/download/0/5/F/05FD6919-6250-425B-86ED-9B095E54065A/wdfcoinstaller.msi wdfcoinstaller.msi 29314207814ce9d5d73695f7e9239539cf37c79e750b9d5ea5a5ef5487a583d6
rm -rf "$WORK/msi" "$WORK/wdk" "$WORK/libwdi-1.5.1"
7z x -y -o"$WORK/msi" "$WORK/wdfcoinstaller.msi" > /dev/null
mkdir -p "$WORK/wdk/redist/wdf/x64" "$WORK/wdk/redist/winusb/x64"
for f in "$WORK"/msi/_WdfCoInstaller01011.dll_* "$WORK"/msi/_winusbcoinstaller2.dll_*; do
  # ai: two of each, x86 and x64, told apart by the PE header's machine (x86-64's objdump format)
  x86_64-w64-mingw32-objdump -f "$f" 2>/dev/null | grep -q 'pei-x86-64' || continue
  case $f in
    *WdfCoInstaller01011*) cp "$f" "$WORK/wdk/redist/wdf/x64/WdfCoInstaller01011.dll" ;;
    *) cp "$f" "$WORK/wdk/redist/winusb/x64/winusbcoinstaller2.dll" ;;
  esac
done
[ -f "$WORK/wdk/redist/wdf/x64/WdfCoInstaller01011.dll" ] && [ -f "$WORK/wdk/redist/winusb/x64/winusbcoinstaller2.dll" ] ||
  { echo "windows-extra: the WDF redistributable holds no x64 co-installers" >&2; exit 1; }
python3 - "$WORK/msi/License.rtf" > "$OUT/licenses/wdf-redistributable.txt" <<'PY'
import re, sys
t = open(sys.argv[1], encoding="latin-1").read()
t = re.sub(r"\\'([0-9a-f]{2})", lambda m: bytes.fromhex(m.group(1)).decode("cp1252"), t)
t = re.sub(r"\\par[d]?\b ?", "\n", t)
t = re.sub(r"\{\\\*[^{}]*\}|\\[a-zA-Z]+-?\d* ?|[{}]", "", t)
t = t[t.find("MICROSOFT SOFTWARE LICENSE TERMS"):]   # ai: the font and style tables before it are no text
print("WinUSB co-installers (WdfCoInstaller01011.dll, winusbcoinstaller2.dll) inside lizard_usb.exe:\n")
print(re.sub(r"\n{3,}", "\n\n", t).strip())
PY
tar xzf "$WORK/libwdi-1.5.1.tar.gz" -C "$WORK"
(
  cd "$WORK/libwdi-1.5.1"
  ./bootstrap.sh > "$WORK/libwdi-bootstrap.log" 2>&1
  ./configure --host=x86_64-w64-mingw32 --disable-32bit --disable-shared --disable-debug --with-wdkdir="$WORK/wdk" \
    --with-wdfver=1011 > "$WORK/libwdi-configure.log" 2>&1
  # ai: configure leaves these two undefined when it cross-builds (its file checks are skipped): the layout above's
  printf '#define COINSTALLER_DIR "winusb"\n#define X64_DIR "x64"\n' >> config.h
  make -j"$(nproc)" > "$WORK/libwdi-make.log" 2>&1
) || { echo "windows-extra: libwdi did not build (logs in $WORK)" >&2; exit 1; }
W=$WORK/libwdi-1.5.1/libwdi
x86_64-w64-mingw32-gcc -O2 -Wall -Wextra -Werror -municode -static -I"$W" "$HERE/../native/usb/lizard_usb.c" "$W/.libs/libwdi.a" \
  -lsetupapi -lcfgmgr32 -lcrypt32 -lole32 -lshell32 -ladvapi32 -lshlwapi -luuid -o "$OUT/lizard_usb.exe"
# ai: the LGPL-3.0, and the GPL-3.0 its text builds on
{ printf 'libwdi 1.5.1 (in lizard_usb.exe; its source: https://github.com/pbatard/libwdi/archive/refs/tags/v1.5.1.tar.gz): COPYING-LGPL, COPYING\n\n'
  cat "$WORK/libwdi-1.5.1/COPYING-LGPL"; printf '\n\n'; cat "$WORK/libwdi-1.5.1/COPYING"; } > "$OUT/licenses/libwdi.txt"
echo "windows-extra: $(ls "$OUT" | grep -c '\.dll$') DLLs, lizard_usb.exe, $(ls "$OUT/licenses" | wc -l) licenses in $OUT"
