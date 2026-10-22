// The licenses the Android app ships under, as one text: the project's license and notice, then every third-party
// component compiled into the APK (the native core, the codec, the fountain, the compression, the fonts).
//   node lizard-android/licenses.mjs [out]      (from the project's root; stdout without out)
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const NOTICES = [
  ["", "LICENSE"],
  ["", "NOTICE"],
  ["Wirehair (liblizard/vendor/wirehair)", "liblizard/vendor/wirehair/LICENSE"],
  ["Zstandard (liblizard/vendor/zstd), under its BSD 3-Clause option", "liblizard/vendor/zstd/LICENSE"],
  ["volk (liblizard/core/third_party/volk)", "liblizard/core/third_party/volk/LICENSE.md"],
  ["JSON for Modern C++ (liblizard/core/third_party/json.hpp)", "liblizard/core/third_party/json.LICENSE.MIT"],
  ["BLAKE3 (liblizard/vendor/blake3), under its Apache License 2.0 option", "liblizard/vendor/blake3/LICENSE_A2"],
  ["Vulkan headers (liblizard/core/third_party/vulkan-headers)", "liblizard/core/third_party/vulkan-headers/LICENSE.md"],
  ["Inter (liblizard/vendor/fonts/inter), under the SIL Open Font License 1.1", "liblizard/vendor/fonts/inter/LICENSE.txt"],
  ["JetBrains Mono (liblizard/vendor/fonts/jetbrains-mono), under the SIL Open Font License 1.1", "liblizard/vendor/fonts/jetbrains-mono/OFL.txt"],
];

const text = NOTICES.map(([t, f]) => (t ? `${t}\n\n` : "") + readFileSync(join(ROOT, f), "utf8").trimEnd()).join(`\n\n${"-".repeat(78)}\n\n`) + "\n";
if (process.argv[2]) writeFileSync(process.argv[2], text);
else process.stdout.write(text);
