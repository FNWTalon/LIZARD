package dev.lizard.desktop

import dev.lizard.pick.Pick
import java.util.Locale
import java.util.prefs.Preferences

// ai: What the sender keeps between runs (the web's send:<id> keys): the encoder (auto, gpu, cpu), the payload (file,
// ai: test), the blocks a frame (0 auto, else 2 to 128), pictures a second, the code's size (25 to 100%), the spacing
// ai: and the margin (modules, Pick's bounds), the ring (-1 the default, else an index into Pick.RINGS, kept as its
// ai: cells), the codes, VSync, the folds and the rail, and whether the brightness tip was taken
// ai: before a first send
// ai: (SendState.requestStart); the receiver's and Home's in a node of their own (Rx, 2026-10-25). forget(): a run that
// ai: neither reads nor keeps either node's, every setting at its default (the test hooks', SenderTest.kt and
// ai: ReceiverTest.kt); the receiver's set in the run held in memory.
object Prefs {
    private var off = false
    private val node by lazy { Preferences.userRoot().node("dev/lizard/sender") }
    private val p get() = if (off) null else node
    fun forget() { off = true }

    // ai: The receiver's (2026-10-25, the web's recv:<id> keys, the Android app's "lizard" preferences): the camera (its
    // ai: id, camera.h Info::id), each camera's own resolution ("WxH") and rate, its zoom a crop at a time (as the app's
    // ai: zoom@<id>/<layout>), its exposure (us) and gain where they were set (none: the camera's own, automatic where it
    // ai: can be), the crop (1:1, 2:1), the decoder (auto, gpu, cpu), the batch size (1 to 32), the
    // ai: folds and the rail, Show statistics; and Home's first-run tips seen. A key is at most Preferences.MAX_KEY_LENGTH (80): a camera's
    // ai: id past 48 characters (a long /dev/v4l/by-id path) is keyed by its hash. Under forget() the node is the run's
    // ai: own, in memory (2026-10-25): nothing kept from before is read and nothing is written, but a setting made in
    // ai: the run holds for the run, as a user's would (a camera's zoom, exposure and gain set again at its next open).
    object Rx {
        private val node by lazy { Preferences.userRoot().node("dev/lizard/receiver") }
        private val mem = java.util.concurrent.ConcurrentHashMap<String, String>()
        // ai: the stored text of a key (Preferences keeps every value as text: putInt's and putBoolean's as toString)
        private fun get(k: String): String? = if (off) mem[k] else node.get(k, null)
        private fun put(k: String, v: String?) {
            if (off) { if (v == null) mem.remove(k) else mem[k] = v } else if (v == null) node.remove(k) else node.put(k, v)
        }
        private fun str(k: String, d: String) = get(k) ?: d
        private fun num(k: String): Double? = get(k)?.toDoubleOrNull()
        private fun putNum(k: String, v: Double?) = put(k, v?.toString())
        private fun int(k: String, d: Int) = get(k)?.trim()?.toIntOrNull() ?: d
        private fun bool(k: String, d: Boolean) = when (get(k)?.lowercase()) { "true" -> true; "false" -> false; else -> d }
        private fun idKey(id: String) = if (id.length <= 48) id else
            java.security.MessageDigest.getInstance("SHA-256").digest(id.toByteArray()).joinToString("") { "%02x".format(it) }.take(32)

        var camera: String get() = str("camera", ""); set(v) { put("camera", v) }
        fun res(cam: String) = str("res@${idKey(cam)}", "")
        fun fps(cam: String) = num("fps@${idKey(cam)}") ?: 0.0
        fun format(cam: String) = str("format@${idKey(cam)}", "")
        fun mode(cam: String, res: String, fps: Double, format: String) {
            put("res@${idKey(cam)}", res); putNum("fps@${idKey(cam)}", fps); put("format@${idKey(cam)}", format)
        }
        fun zoom(cam: String, layout: String) = num("zoom@${idKey(cam)}/$layout")
        fun zoom(cam: String, layout: String, v: Double) = putNum("zoom@${idKey(cam)}/$layout", v)
        // ai: null: the camera's own (automatic where it can be)
        fun exposure(cam: String) = num("exposure@${idKey(cam)}")
        fun exposure(cam: String, us: Double?) = putNum("exposure@${idKey(cam)}", us)
        fun gain(cam: String) = num("gain@${idKey(cam)}")
        fun gain(cam: String, v: Double?) = putNum("gain@${idKey(cam)}", v)
        var layout: String get() = str("layout", "1:1").let { if (it == "2:1") it else "1:1" }; set(v) { put("layout", v) }
        var decoder: String get() = str("decoder", "auto").let { if (it == "gpu" || it == "cpu") it else "auto" }; set(v) { put("decoder", v) }
        var batch: Int get() = int("batch", 32).coerceIn(1, 32); set(v) { put("batch", v.toString()) }
        var settingsOpen: Boolean get() = bool("recvSettings", false); set(v) { put("recvSettings", v.toString()) }
        var toolsOpen: Boolean get() = bool("recvTools", false); set(v) { put("recvTools", v.toString()) }
        var collapsed: Boolean get() = bool("recvCollapsed", false); set(v) { put("recvCollapsed", v.toString()) }
        var stats: Boolean get() = bool("stats", false); set(v) { put("stats", v.toString()) }
        var tipsSeen: Boolean get() = bool("tipsSeen", false); set(v) { put("tipsSeen", v.toString()) }
    }
    private fun str(k: String, d: String) = p?.get(k, d) ?: d
    private fun int(k: String, d: Int) = p?.getInt(k, d) ?: d
    private fun bool(k: String, d: Boolean) = p?.getBoolean(k, d) ?: d

    var enc: String get() = str("enc", "auto"); set(v) { p?.put("enc", v) }
    var payload: String get() = str("payload", "file"); set(v) { p?.put("payload", v) }
    var blocks: Int get() = int("blocks", 60); set(v) { p?.putInt("blocks", v) }   // ai: the page's LIZARD-480 default
    var fps: Int get() = int("fps", 60); set(v) { p?.putInt("fps", v) }
    var size: Int get() = int("size", 100); set(v) { p?.putInt("size", v) }
    var margin: Int get() = int("margin", Room.MARGIN_DEFAULT); set(v) { p?.putInt("margin", v) }
    var spacing: Int get() = int("spacing", Room.SPACING_DEFAULT); set(v) { p?.putInt("spacing", v) }
    // ai: an index into Pick.RINGS, kept by its cells (ringCells), so a change of the list moves no saved choice: cells
    // ai: no ring has any more read as the default
    var ring: Int
        get() = Pick.RINGS.indexOf(int("ringCells", 0))
        set(v) { p?.putInt("ringCells", if (v in Pick.RINGS.indices) Pick.RINGS[v] else -1) }
    var codes: Int get() = int("codes", 1); set(v) { p?.putInt("codes", v) }
    var vsync: Boolean get() = bool("vsync", true); set(v) { p?.putBoolean("vsync", v) }   // ai: on unless turned off
    var settingsOpen: Boolean get() = bool("dev", false); set(v) { p?.putBoolean("dev", v) }
    var toolsOpen: Boolean get() = bool("logs", false); set(v) { p?.putBoolean("logs", v) }
    var collapsed: Boolean get() = bool("collapsed", false); set(v) { p?.putBoolean("collapsed", v) }
    // ai: the first-send tip taken (Screen.kt BrightnessTip); "tipSeen" since 2026-10-23, when it gained power saving
    // ai: and was due once more ("brightnessSeen" before)
    var brightnessSeen: Boolean get() = bool("tipSeen", false); set(v) { p?.putBoolean("tipSeen", v) }
}

// ai: The sender's own figures beside Readout's (liblizard/kotlin/Readout.kt bytes and rate): digits as the web prints
// ai: them whatever the locale, a figure kept on its line (a no-break space).
object Fmt {
    fun blocks(b: Int) = "$b, ${String.format(Locale.ROOT, "%.1f", b * 469 / 1000.0)}\u00a0KB"   // ai: under the title "Blocks" (2026-10-05)
    fun f1(v: Double) = String.format(Locale.ROOT, "%.1f", v)
}
