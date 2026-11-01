package dev.lizard.desktop

import dev.lizard.pick.Pick
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.util.Locale
import java.util.prefs.Preferences

// ai: What the sender keeps between runs (the web's send:<id> keys): the encoder (auto, gpu, cpu), the payload (file,
// ai: test), the format's size (0 auto, else a version 1 to 64), pictures a second, the code's size (25 to 100%), the spacing
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
    // ai: a run that keeps nothing (the test hooks'): the receive session's camera settings held in memory too
    val forgotten get() = off

    // ai: The receiver's (2026-10-25, the web's recv:<id> keys, the Android app's "lizard" preferences): the camera (its
    // ai: id, camera.h Info::id), the crop (1:1, 2:1), the decoder (auto, gpu, cpu), the batch size (1 to 32), the folds
    // ai: and the rail, Show statistics; and Home's first-run tips seen. Each camera's own mode, zoom, exposure and gain
    // ai: are the receive session's since 2026-10-31 (Native.data/cameras.json, liblizard/core/rx/session.h), taken over
    // ai: once from the keys this node held for them (camerasMoved). Under forget() the node is the run's own, in memory
    // ai: (2026-10-25): nothing kept from before is read and nothing is written, but a setting made in the run holds for
    // ai: the run, as a user's would.
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
        private fun int(k: String, d: Int) = get(k)?.trim()?.toIntOrNull() ?: d
        private fun bool(k: String, d: Boolean) = when (get(k)?.lowercase()) { "true" -> true; "false" -> false; else -> d }
        // ai: a camera's id as these keys held it (at most Preferences.MAX_KEY_LENGTH, 80: an id past 48 characters, a
        // ai: long /dev/v4l/by-id path, by its hash)
        private fun idKey(id: String) = if (id.length <= 48) id else
            java.security.MessageDigest.getInstance("SHA-256").digest(id.toByteArray()).joinToString("") { "%02x".format(it) }.take(32)

        var camera: String get() = str("camera", ""); set(v) { put("camera", v) }

        // ai: The cameras' settings this node held before 2026-10-31 (res@, fps@, format@, zoom@<id>/<layout>, exposure@,
        // ai: gain@) in the session's cameras.json form (session.h), for the one move into it: {"version": 1, "cameras":
        // ai: {"<id>": {"mode", "zoom", "exposure", "gain"}}}; null where the node held none. A camera keyed by its id's
        // ai: hash (an id past 48 characters: V4L2's by-id paths, Media Foundation's symbolic links, nearly every UVC
        // ai: camera) is found among the camera chosen and the ids listed (asked only where such a key is: a listing opens
        // ai: devices); one whose camera is not plugged in is left behind. The keys stay, unread.
        fun camerasMoved(listed: () -> List<String>): String? {
            if (off) return null
            val keys = runCatching { node.keys().toList() }.getOrDefault(emptyList())
            fun hashed(k: String) = k.length == 32 && k.all { it in "0123456789abcdef" }
            val chosen = camera
            val known = HashMap<String, String>()
            if (chosen.isNotEmpty()) known[idKey(chosen)] = chosen
            if (keys.any { k -> k.indexOf('@') > 0 && hashed(k.substringAfter('@').substringBeforeLast('/')) && k.substringAfter('@').substringBeforeLast('/') !in known })
                for (id in runCatching(listed).getOrDefault(emptyList())) if (id.length > 48) known.getOrPut(idKey(id)) { id }
            fun idOf(k: String): String? = if (k.length <= 48 && !hashed(k)) k else known[k]
            val out = LinkedHashMap<String, MutableMap<String, JsonElement>>()
            fun at(id: String) = out.getOrPut(id) { LinkedHashMap() }
            for (k in keys) {
                val at = k.indexOf('@')
                if (at <= 0) continue
                val what = k.substring(0, at)
                val rest = k.substring(at + 1)
                when (what) {
                    "res" -> {
                        val id = idOf(rest) ?: continue
                        val wh = str(k, "").split('x').mapNotNull { it.trim().toIntOrNull() }
                        if (wh.size != 2) continue
                        at(id)["mode"] = JsonObject(mapOf(
                            "w" to JsonPrimitive(wh[0]), "h" to JsonPrimitive(wh[1]),
                            "fps" to JsonPrimitive(num("fps@$rest") ?: 0.0),
                            "format" to JsonPrimitive(str("format@$rest", ""))))
                    }
                    "zoom" -> {
                        val slash = rest.lastIndexOf('/')
                        if (slash <= 0) continue
                        val id = idOf(rest.substring(0, slash)) ?: continue
                        val v = num(k) ?: continue
                        val z = (at(id)["zoom"] as? JsonObject)?.toMutableMap() ?: LinkedHashMap()
                        z[rest.substring(slash + 1)] = JsonPrimitive(v)
                        at(id)["zoom"] = JsonObject(z)
                    }
                    "exposure", "gain" -> {
                        val id = idOf(rest) ?: continue
                        at(id)[what] = JsonPrimitive(num(k) ?: continue)
                    }
                }
            }
            if (out.isEmpty()) return null
            return JsonObject(mapOf("version" to JsonPrimitive(1),
                "cameras" to JsonObject(out.mapValues { JsonObject(it.value) }))).toString()
        }
        var layout: String get() = str("layout", "1:1").let { if (it == "2:1") it else "1:1" }; set(v) { put("layout", v) }
        var decoder: String get() = str("decoder", "auto").let { if (it == "gpu" || it == "cpu") it else "auto" }; set(v) { put("decoder", v) }
        var batch: Int get() = int("batch", 32).coerceIn(1, 32); set(v) { put("batch", v.toString()) }
        var experimentalOpen: Boolean get() = bool("recvExperimental", false); set(v) { put("recvExperimental", v.toString()) }
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
    // ai: kept as "version" since the versions went to 16 sub-channels (2026-10-29; "blocks" before counted 8)
    var blocks: Int get() = int("version", 30); set(v) { p?.putInt("version", v) }   // ai: the page's LIZARD-480 default
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
    // ai: Exclusive fullscreen (Send's Experimental fold, 2026-10-31): Fullscreen while sending takes the monitor (SendState.lease)
    var exclusiveFull: Boolean get() = bool("exclusiveFull", false); set(v) { p?.putBoolean("exclusiveFull", v) }
    // ai: its warning taken (Screen.kt ExclusiveTip): shown at the first turn-on until it is accepted
    var exclusiveSeen: Boolean get() = bool("exclusiveSeen", false); set(v) { p?.putBoolean("exclusiveSeen", v) }
    var experimentalOpen: Boolean get() = bool("experimental", false); set(v) { p?.putBoolean("experimental", v) }
    var vsync: Boolean get() = bool("vsync", true); set(v) { p?.putBoolean("vsync", v) }   // ai: on unless turned off
    var settingsOpen: Boolean get() = bool("dev", false); set(v) { p?.putBoolean("dev", v) }
    var toolsOpen: Boolean get() = bool("logs", false); set(v) { p?.putBoolean("logs", v) }
    var collapsed: Boolean get() = bool("collapsed", false); set(v) { p?.putBoolean("collapsed", v) }
    // ai: the tool screens' column as its edge was dragged, dp (2026-10-31; 0: the web's 20rem), Receive's and Send's
    var sideDp: Float get() = p?.getFloat("sideDp", 0f) ?: 0f; set(v) { p?.putFloat("sideDp", v) }
    // ai: the first-send tip taken (Screen.kt BrightnessTip); "tipSeen" since 2026-10-23, when it gained power saving
    // ai: and was due once more ("brightnessSeen" before)
    var brightnessSeen: Boolean get() = bool("tipSeen", false); set(v) { p?.putBoolean("tipSeen", v) }
}

// ai: The sender's own figures beside Readout's (liblizard/kotlin/Readout.kt bytes and rate): digits as the web prints
// ai: them whatever the locale, a figure kept on its line (a no-break space).
object Fmt {
    fun blocks(b: Int) = "$b, ${String.format(Locale.ROOT, "%.1f", b * Pick.PAYLOAD / 1000.0)}\u00a0KB"   // ai: under the title "Blocks" (2026-10-05)
    fun f1(v: Double) = String.format(Locale.ROOT, "%.1f", v)
}
