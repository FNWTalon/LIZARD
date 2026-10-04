#!/bin/sh
# ai: The desktop sender's .deb (build.gradle.kts packageDeb, then this as debDeps): jpackage lists in Depends what the
# ai: bundled Java runtime and Compose's natives link; the app's own native library (inside its jar, extracted at start)
# ai: also needs libvulkan1 (volk opens libvulkan.so.1: the presenter draws only through Vulkan) and libxrandr2 (the
# ai: monitor's refresh), added here. Each .deb in the folder given, once (2026-10-04).
set -e
for deb in "$1"/*.deb; do
  [ -f "$deb" ] || continue
  t=$(mktemp -d)
  dpkg-deb -R "$deb" "$t/x"
  if ! grep -q 'libvulkan1' "$t/x/DEBIAN/control"; then
    sed -i '/^Depends:/ s/[[:space:]]*$/, libvulkan1, libxrandr2/' "$t/x/DEBIAN/control"
    fakeroot dpkg-deb -b --root-owner-group "$t/x" "$deb" > /dev/null
  fi
  rm -rf "$t"
done
