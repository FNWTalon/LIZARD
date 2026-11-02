#!/bin/sh
# ai: The desktop app's .deb (build.gradle.kts packageDeb, then this as debDeps): jpackage lists in Depends what the
# ai: bundled Java runtime and Compose's natives link; the app's own native library (inside its jar, extracted at start)
# ai: also needs libvulkan1 (volk opens libvulkan.so.1: the presenter draws only through Vulkan), libxrandr2 (the
# ai: monitor's refresh), and since the receiver (2026-10-25) TurboJPEG for an MJPEG camera's frames (liblizard/core/cam/jpeg.cpp
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
  # ai: since 2026-11-01: libusb for the phone on the cable (liblizard/core/cam/aoa.cpp opens libusb-1.0.so.0) and Aravis
  # ai: recommended for a GenICam camera (cam/aravis.cpp opens libaravis-0.8.so.0); 70-lizard.rules gives the user at the
  # ai: seat those devices, the rules read again at the install
  if ! grep -q 'libusb-1.0-0' "$t/x/DEBIAN/control"; then
    sed -i '/^Depends:/ s/[[:space:]]*$/, libusb-1.0-0/' "$t/x/DEBIAN/control"
    changed=1
  fi
  if ! grep -q '^Recommends:.*libaravis' "$t/x/DEBIAN/control"; then
    if grep -q '^Recommends:' "$t/x/DEBIAN/control"; then sed -i '/^Recommends:/ s/[[:space:]]*$/, libaravis-0.8-0/' "$t/x/DEBIAN/control"
    else sed -i '/^Depends:/ a Recommends: libaravis-0.8-0' "$t/x/DEBIAN/control"; fi
    changed=1
  fi
  if [ ! -f "$t/x/lib/udev/rules.d/70-lizard.rules" ]; then
    install -D -m 644 "$(dirname "$0")/70-lizard.rules" "$t/x/lib/udev/rules.d/70-lizard.rules"
    for s in postinst postrm; do
      f="$t/x/DEBIAN/$s"
      [ -f "$f" ] || printf '#!/bin/sh\nset -e\n' > "$f"
      # ai: before the script's own exit (jpackage's end in "exit 0"), else at its end
      line='if command -v udevadm > /dev/null 2>&1; then udevadm control --reload-rules || true; udevadm trigger --subsystem-match=usb --action=change || true; fi'
      if grep -q '^exit 0' "$f"; then
        awk -v l="$line" '!done && /^exit 0/ { print l; done = 1 } { print }' "$f" > "$f.new" && mv "$f.new" "$f"
      else printf '%s\n' "$line" >> "$f"; fi
      chmod 755 "$f"
    done
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
