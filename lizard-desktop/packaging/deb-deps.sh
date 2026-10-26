#!/bin/sh
# ai: The desktop app's .deb (build.gradle.kts packageDeb, then this as debDeps): jpackage lists in Depends what the
# ai: bundled Java runtime and Compose's natives link; the app's own native library (inside its jar, extracted at start)
# ai: also needs libvulkan1 (volk opens libvulkan.so.1: the presenter draws only through Vulkan), libxrandr2 (the
# ai: monitor's refresh), and since the receiver (2026-10-25) TurboJPEG for an MJPEG camera's frames (camera/jpeg.cpp
# ai: opens libturbojpeg.so.0 at run time): libturbojpeg, Ubuntu's name for it, or libturbojpeg0, Debian's; added here
# ai: where missing. Each .deb in the folder given, once (2026-10-04). Since 2026-10-23 its desktop entry also takes
# ai: files and folders (Exec ... %F, the sender adding them: Main.kt) and lists the sender for folders
# ai: (MimeType=inode/directory;), so a file manager's Open with hands them over; files go through its Other
# ai: Application, which the manager then keeps (listing application/octet-stream would make the sender the opener of
# ai: every unknown file where nothing else claims it).
set -e
for deb in "$1"/*.deb; do
  [ -f "$deb" ] || continue
  t=$(mktemp -d)
  dpkg-deb -R "$deb" "$t/x"
  changed=0
  for dep in libvulkan1 libxrandr2; do
    if ! grep -q "$dep" "$t/x/DEBIAN/control"; then
      sed -i "/^Depends:/ s/[[:space:]]*\$/, $dep/" "$t/x/DEBIAN/control"
      changed=1
    fi
  done
  if ! grep -q 'libturbojpeg' "$t/x/DEBIAN/control"; then
    sed -i '/^Depends:/ s/[[:space:]]*$/, libturbojpeg | libturbojpeg0/' "$t/x/DEBIAN/control"
    changed=1
  fi
  for entry in "$t"/x/opt/*/lib/*.desktop; do
    [ -f "$entry" ] || continue
    if ! grep -q '^Exec=.*%F' "$entry"; then
      sed -i -e 's|^Exec=\(.*\)$|Exec=\1 %F|' -e 's|^MimeType=.*$|MimeType=inode/directory;|' "$entry"
      changed=1
    fi
  done
  [ "$changed" = 1 ] && fakeroot dpkg-deb -b --root-owner-group "$t/x" "$deb" > /dev/null
  rm -rf "$t"
done
