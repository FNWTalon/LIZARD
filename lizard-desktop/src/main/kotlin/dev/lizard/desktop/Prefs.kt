package dev.lizard.desktop

import java.util.Locale
import java.util.prefs.Preferences
import kotlin.math.roundToInt

// ai: What the sender keeps between runs (the web's send:<id> keys): the encoder (auto, gpu, cpu), the payload (file,
// ai: test), the blocks a frame (0 auto, else 1 to 128), pictures a second, the code's size (25 to 100%), the gap between
// ai: two codes (0 to 64 modules), the ring (-1 the default, else 0 to 3), the codes, the folds and the rail, and
// ai: whether the brightness tip was taken before a first send (SendState.requestStart). forget():
// ai: a run that neither reads nor keeps them, every setting at its default (the test hook's, SenderTest.kt).
object Prefs {
    private var off = false
    private val node by lazy { Preferences.userRoot().node("dev/lizard/sender") }
    private val p get() = if (off) null else node
    fun forget() { off = true }
    private fun str(k: String, d: String) = p?.get(k, d) ?: d
    private fun int(k: String, d: Int) = p?.getInt(k, d) ?: d
    private fun bool(k: String, d: Boolean) = p?.getBoolean(k, d) ?: d

    var enc: String get() = str("enc", "auto"); set(v) { p?.put("enc", v) }
    var payload: String get() = str("payload", "file"); set(v) { p?.put("payload", v) }
    var blocks: Int get() = int("blocks", 60); set(v) { p?.putInt("blocks", v) }   // ai: the page's LIZARD-480 default
    var fps: Int get() = int("fps", 60); set(v) { p?.putInt("fps", v) }
    var size: Int get() = int("size", 100); set(v) { p?.putInt("size", v) }
    var gap: Int get() = int("gap", Pick.GAP_DEFAULT); set(v) { p?.putInt("gap", v) }
    var ring: Int get() = int("ring", -1); set(v) { p?.putInt("ring", v) }
    var codes: Int get() = int("codes", 1); set(v) { p?.putInt("codes", v) }
    var settingsOpen: Boolean get() = bool("dev", false); set(v) { p?.putBoolean("dev", v) }
    var toolsOpen: Boolean get() = bool("logs", false); set(v) { p?.putBoolean("logs", v) }
    var collapsed: Boolean get() = bool("collapsed", false); set(v) { p?.putBoolean("collapsed", v) }
    var brightnessSeen: Boolean get() = bool("brightnessSeen", false); set(v) { p?.putBoolean("brightnessSeen", v) }
}

// ai: The web's figures (lizard-web/ui.mjs bytes, rate; lizard-android/.../Readout.kt the same): 1000 B a KB, the unit chosen
// ai: after rounding, a figure kept on its line (a no-break space), digits as the web prints them whatever the locale.
object Fmt {
    const val NB = '\u00a0'
    private fun at(v: Double) = if (v < 9.95) String.format(Locale.ROOT, "%.1f", v) else v.roundToInt().toString()
    fun bytes(n: Long): String {
        if (n < 1000) return "$n${NB}B"
        return when {
            n < 999_500 -> "${at(n / 1e3)}${NB}KB"
            n < 999_500_000 -> "${at(n / 1e6)}${NB}MB"
            else -> "${at(n / 1e9)}${NB}GB"
        }
    }
    fun rate(kbs: Double) = if (kbs < 999.5) "${kbs.roundToInt()}${NB}KB/s" else "${String.format(Locale.ROOT, "%.2f", kbs / 1000)}${NB}MB/s"
    fun blocks(b: Int) = "$b, ${String.format(Locale.ROOT, "%.1f", b * 469 / 1000.0)}${NB}KB"   // ai: under the title "Blocks" (2026-10-05)
    fun f1(v: Double) = String.format(Locale.ROOT, "%.1f", v)
}
