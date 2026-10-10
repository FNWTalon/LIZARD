# Fonts: Inter 4.1 and JetBrains Mono 2.304

The two typefaces every LIZARD surface draws its text in, as released, nothing changed:

- `inter/`: Inter 4.1 (https://github.com/rsms/inter, release v4.1, `Inter-4.1.zip`, sha256
  9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e). `InterVariable.ttf` and `web/InterVariable.woff2`
  from the archive: one variable font with every weight, the interface's text. SIL Open Font License 1.1
  (`LICENSE.txt`).
- `jetbrains-mono/`: JetBrains Mono 2.304 (https://github.com/JetBrains/JetBrainsMono, release v2.304,
  `JetBrainsMono-2.304.zip`, sha256 6f6376c6ed2960ea8a963cd7387ec9d76e3f629125bc33d1fdcd7eb7012f7bbf).
  `fonts/ttf/JetBrainsMono-Regular.ttf` and `fonts/webfonts/JetBrainsMono-Regular.woff2`: the readouts' monospace. SIL
  Open Font License 1.1 (`OFL.txt`, `AUTHORS.txt`).

Neither license reserves a font name. The web pages load the `.woff2` files (`lizard-web/ui.css`); the Android app and
the desktop sender carry the `.ttf` files, copied in at build time (`lizard-android/app/build.gradle.kts`,
`lizard-desktop/build.gradle.kts`), Inter at the weights 400, 500, 600 and 700 through its weight axis.
