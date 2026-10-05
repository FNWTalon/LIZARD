package dev.lizard.receiver

import android.content.Context

// ai: The app's switches, in SharedPreferences "lizard"; the decoder in Settings, the rest in
// ai: Advanced:
// ai:   decoder     auto | gpu | cpu                      ReceiverConfig.decoder
// ai:   precision   auto | int8 | f16 | f32               ReceiverConfig.precision; set only by tools/phone/ab.sh p= since
// ai:               2026-10-01 (its chips went)
// ai:   layout      1:1 | 2:1                            ReceiverConfig.layout (2:1: the frame's centre 2:1, cut in half)
// ai:   camera      auto (the back camera that focuses closest, Engine.rearId) or a back camera's id (Engine.lenses)
// ai:   resolution  1280x720 | 1920x1080 | 2560x1440 | 3840x2160, the camera's ImageReader (1920x1080)
// ai:   zoom        the camera's zoom ratio, 0.1 apart over its range up to 4 (Advanced's slider since 2026-10-01; chips
// ai:               of 1, 1.4 and 2 before); 1.5 by default since 2026-10-02 (the zoom of the 2:1 runs at 2.2+ MB/s;
// ai:               1.4 before, the 2026-09-30 sweep's best, 1.7 next): the code in the middle of the lens's field, not
// ai:               out to its soft corners (Engine.session; set live, Engine.zoom)
// ai:   focus       auto | dioptres (0 infinity, to the lens's closest): Receive's Focus (2026-10-04): auto the
// ai:               camera's continuous video autofocus, else the lens held there (Engine.applyFocus; set live, Engine.focus)
// ai:   devlog      the rig's address (http://host:8080), empty for none
// ai:   batch       1 to 32, the most frames a GPU batch waits for (Receive's Settings, 2026-10-02: the lock's readings
// ai:               come back a batch's wait after their capture, so at 1 the lock can re-tune each frame); 32 by
// ai:               default, the batcher's own size (receiver.h batchCap); set live
// ai:   phase       off | track: the camera's phase against the display (PhaseLock.kt; track by default since
// ai:               2026-10-01, the night it read 93 to 95% of a sender at 60
// ai:               painted; a sender that paints slower loses nothing to it; a stored "auto", the arm deleted that
// ai:               day, loads as track). `debug.lizard.phase`, when set, overrides it (the tools set it).
// ai:   replays     off | on: Save replays (Developer Tools, 2026-10-03; MainActivity's replay): the newest frames the
// ai:               decoder is handed kept while the camera runs; off by default, to spare storage. Since 2026-10-05 the
// ai:               switch is the session's alone, never saved (a recording left on costs every later run its lag):
// ai:               every start begins off, and a stored "on" from before is removed at the next save
// ai: The camera's rate is no switch since 2026-10-01: [60,60], else the highest fixed range (Engine.pickFps).
// ai: Resolution, zoom and focus are a lens's (2026-10-04): each is kept as "<key>@<camera id>" for the back camera the
// ai: camera switch resolves to (Engine.rearId: auto is the closest-focusing lens, so auto and that lens share one set),
// ai: and choosing another lens brings that lens's own three back (MainActivity.change). A lens with none saved reads
// ai: the plain key (the one value every lens shared before), else the default; a save writes the lens's keys and drops
// ai: the plain ones, so tools/phone/ab.sh, which cannot name the lens, writes the plain zoom and removes the "zoom@"
// ai: entries, and the app adopts it for the lens it opens. The other switches are the app's, under their plain keys.
data class Settings(
    val decoder: String = "auto",
    val precision: String = "auto",
    val layout: String = "1:1",
    val camera: String = "auto",
    val resolution: String = "1920x1080",
    val zoom: String = "1.5",
    val devlog: String = "",
    val phase: String = "track",
    val batch: String = "32",
    val replays: String = "off",
    val focus: String = "auto",
) {
    val frames get() = batch.toIntOrNull()?.coerceIn(1, 32) ?: 32

    // ai: cam: the id the camera switch resolves to (Engine.rearId), null where the phone has no back camera (the
    // ai: lens's values then stay under the plain keys)
    fun save(ctx: Context, cam: String?) {
        val e = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString("decoder", decoder).putString("precision", precision).putString("layout", layout).putString("camera", camera)
            .remove("fps").putString("devlog", devlog).putString("phase", phase).putString("batch", batch).remove("replays")
        for ((k, v) in lens()) { e.putString(key(k, cam), v); if (cam != null) e.remove(k) }
        e.apply()
    }

    // ai: this, with the lens's three as saved for cam (else the plain key's, else the defaults)
    fun forCamera(ctx: Context, cam: String?): Settings {
        val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val d = Settings()
        fun g(k: String, dflt: String) = p.getString(key(k, cam), null) ?: p.getString(k, null) ?: dflt
        return copy(resolution = g("resolution", d.resolution), zoom = g("zoom", d.zoom), focus = g("focus", d.focus))
    }

    private fun lens() = listOf("resolution" to resolution, "zoom" to zoom, "focus" to focus)

    companion object {
        const val PREFS = "lizard"
        val DECODERS = listOf("auto", "gpu", "cpu")
        val PRECISIONS = listOf("auto", "int8", "f16", "f32")
        val LAYOUTS = listOf("1:1", "2:1")
        val RESOLUTIONS = listOf("1280x720", "1920x1080", "2560x1440", "3840x2160")
        val PHASES = listOf("off", "track")

        private fun key(k: String, cam: String?) = if (cam == null) k else "$k@$cam"

        // ai: rearId: the camera switch's value to the back camera it opens (Engine.rearId)
        fun load(ctx: Context, rearId: (String) -> String?): Settings {
            val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val d = Settings()
            val s = Settings(p.getString("decoder", d.decoder)!!, p.getString("precision", d.precision)!!,
                p.getString("layout", d.layout)!!, p.getString("camera", d.camera)!!, d.resolution, d.zoom, p.getString("devlog", d.devlog)!!,
                p.getString("phase", d.phase)!!.let { if (it == "auto") "track" else it }, p.getString("batch", d.batch)!!, d.replays,
                d.focus)
            return s.forCamera(ctx, rearId(s.camera))
        }
    }
}
