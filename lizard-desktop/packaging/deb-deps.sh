#!/bin/sh
# ai: The desktop sender's .deb (build.gradle.kts packageDeb, then this as debDeps): jpackage lists in Depends what the
# ai: bundled Java runtime and Compose's natives link; the app's own native library (inside its jar, extracted at start)
# ai: also needs libvulkan1 (volk opens libvulkan.so.1: the presenter draws only through Vulkan), libxrandr2 (the
# ai: monitor's refresh), libxcomposite1 and libxext6 (whether a compositor draws the window, 2026-10-07), added here
# ai: where missing. Each .deb in the folder given, once (2026-10-04).
set -e
for deb in "$1"/*.deb; do
  [ -f "$deb" ] || continue
  t=$(mktemp -d)
  dpkg-deb -R "$deb" "$t/x"
  changed=0
  for dep in libvulkan1 libxrandr2 libxcomposite1 libxext6; do
    if ! grep -q "$dep" "$t/x/DEBIAN/control"; then
      sed -i "/^Depends:/ s/[[:space:]]*\$/, $dep/" "$t/x/DEBIAN/control"
      changed=1
    fi
  done
  [ "$changed" = 1 ] && fakeroot dpkg-deb -b --root-owner-group "$t/x" "$deb" > /dev/null
  rm -rf "$t"
done
