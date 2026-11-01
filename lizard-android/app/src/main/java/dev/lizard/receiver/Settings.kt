package dev.lizard.receiver

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import org.json.JSONObject

// ai: The app's switches, in SharedPreferences "lizard"; Receive's Settings but replays and stats, which are Developer
// ai: Tools':
// ai:   decoder     auto | gpu | cpu | pc                 the receive session's decoder (session.h; pc the PC link's)
// ai:   layout      1:1 | 2:1                            its crop (2:1: the frame's centre 2:1, cut in half)
// ai:   camera      a back camera's id (Settings' Camera menu, the session's list); empty, or one the phone no longer
// ai:               lists, opens the back camera that focuses closest (the session's first listed)
// ai:   batch       1 to 32, the most frames a GPU launch takes (Receive's Settings, 2026-10-02; since 2026-10-08 a
// ai:               launch goes as soon as a frame waits and takes every frame then waiting up to this); 32 by default
// ai:               (receiver.h batchCap); set live
// ai:   replays     off | on: Save replays (Developer Tools, 2026-10-03; the session's): the newest frames the decoder is
// ai:               handed kept while the camera runs; never saved (a recording left on costs every later run its lag):
// ai:               every start begins off
// ai:   stats       on | off: Show statistics (Developer Tools, 2026-10-10): off hides Receive's green line once a file
// ai:               is in and Developer Tools' lab line and readout; off by default, kept
// ai:   pcPack      on | off: the PC decoder's frames packed as .lizard frames (the default) or sent raw (a phone too weak
// ai:               or too warm to pack them)
// ai: A camera's own settings are the receive session's since 2026-10-31, kept per camera in <data>/cameras.json: its
// ai: mode (Resolution and Format; the rate is the camera's fixed one, [60,60] where offered) and its zoom in each crop
// ai: (1.5 unless set). Before that they were keys here (resolution@<id>, format@<id>, zoom@<id>/<layout>), moved into
// ai: cameras.json once (camerasMoved) and left in place, unread.
data class Settings(
    val decoder: String = "auto",
    val layout: String = "1:1",
    val camera: String = "",
    val batch: String = "32",
    val replays: String = "off",
    val stats: String = "off",
    val pcPack: String = "on",
) {
    val frames get() = batch.toIntOrNull()?.coerceIn(1, 32) ?: 32
    fun save(ctx: Context) {
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString("decoder", decoder).putString("layout", layout).putString("camera", camera)
            .putString("batch", batch).putString("stats", stats).putString("pcPack", pcPack).apply()
    }

    companion object {
        const val PREFS = "lizard"
        val LAYOUTS = listOf("1:1", "2:1")
        val RESOLUTIONS = listOf("1280x720", "1920x1080", "2560x1440", "3840x2160")
        // ai: the formats the app reads, the menu's order and labels (the camera's modes say which the lens offers)
        val FORMATS = listOf("private" to "Private", "yuv" to "YUV", "y8" to "Y8", "raw16" to "RAW16", "raw10" to "RAW10", "raw12" to "RAW12")

        fun load(ctx: Context): Settings {
            val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val d = Settings()
            return Settings(p.getString("decoder", d.decoder)!!, p.getString("layout", d.layout)!!, p.getString("camera", d.camera)!!,
                p.getString("batch", d.batch)!!, d.replays, p.getString("stats", d.stats)!!, p.getString("pcPack", d.pcPack)!!)
        }

        // ai: The cameras' settings the app kept before the session kept them, as cameras.json (session.h's header):
        // ai: "android:<id>": a mode where a resolution was kept ({w, h, fps, format}: fps the camera's own rate, as its
        // ai: modes name it; a format the session's camera no longer offers, P010, as auto), the zoom of each crop kept.
        // ai: Null where none is kept.
        fun camerasMoved(ctx: Context): String? {
            val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).all
            val ids = p.keys.mapNotNull { k ->
                when {
                    k.startsWith("resolution@") -> k.removePrefix("resolution@")
                    k.startsWith("format@") -> k.removePrefix("format@")
                    k.startsWith("zoom@") -> k.removePrefix("zoom@").substringBefore('/')
                    else -> null
                }
            }.toSet()
            if (ids.isEmpty()) return null
            val cm = ctx.getSystemService(CameraManager::class.java)
            val cams = JSONObject()
            for (id in ids) {
                val e = JSONObject()
                val res = (p["resolution@$id"] as? String)?.split("x")?.mapNotNull { it.toIntOrNull() }?.takeIf { it.size == 2 }
                val fmt = (p["format@$id"] as? String)?.takeIf { f -> f == "auto" || FORMATS.any { it.first == f } } ?: "auto"
                val fps = runCatching { fpsOf(cm.getCameraCharacteristics(id)) }.getOrNull()
                if (fps != null && (res != null || fmt != "auto")) {
                    val (w, h) = res ?: listOf(1920, 1080)
                    e.put("mode", JSONObject().put("w", w).put("h", h).put("fps", fps).put("format", fmt))
                }
                val z = JSONObject()
                for (l in LAYOUTS) (p["zoom@$id/$l"] as? String)?.toDoubleOrNull()?.let { z.put(l, it) }
                if (z.length() > 0) e.put("zoom", z)
                if (e.length() > 0) cams.put("android:$id", e)
            }
            return if (cams.length() == 0) null else JSONObject().put("version", 1).put("cameras", cams).toString(1)
        }

        // ai: the rate the session's camera runs a lens at (cam/android.cpp pickFps): [60, 60] where offered, else the
        // ai: highest fixed range, else the highest top
        private fun fpsOf(ch: CameraCharacteristics): Int? {
            val r = ch.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.toList().orEmpty()
            if (r.isEmpty()) return null
            if (r.any { it.lower == 60 && it.upper == 60 }) return 60
            return (r.filter { it.lower == it.upper }.maxByOrNull { it.upper } ?: r.maxBy { it.upper }).upper
        }
    }
}
