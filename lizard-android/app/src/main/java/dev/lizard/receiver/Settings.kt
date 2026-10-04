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
// ai:               decoder is handed kept while the camera runs; off by default, to spare storage
// ai: The camera's rate is no switch since 2026-10-01: [60,60], else the highest fixed range (Engine.pickFps).
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

    fun save(ctx: Context) {
        ctx.getSharedPreferences("lizard", Context.MODE_PRIVATE).edit()
            .putString("decoder", decoder).putString("precision", precision).putString("layout", layout).putString("camera", camera).putString("resolution", resolution)
            .remove("fps").putString("zoom", zoom).putString("devlog", devlog).putString("phase", phase).putString("batch", batch).putString("replays", replays).putString("focus", focus).apply()
    }

    companion object {
        val DECODERS = listOf("auto", "gpu", "cpu")
        val PRECISIONS = listOf("auto", "int8", "f16", "f32")
        val LAYOUTS = listOf("1:1", "2:1")
        val RESOLUTIONS = listOf("1280x720", "1920x1080", "2560x1440", "3840x2160")
        val PHASES = listOf("off", "track")

        fun load(ctx: Context): Settings {
            val p = ctx.getSharedPreferences("lizard", Context.MODE_PRIVATE)
            val d = Settings()
            return Settings(p.getString("decoder", d.decoder)!!, p.getString("precision", d.precision)!!,
                p.getString("layout", d.layout)!!, p.getString("camera", d.camera)!!, p.getString("resolution", d.resolution)!!, p.getString("zoom", d.zoom)!!, p.getString("devlog", d.devlog)!!,
                p.getString("phase", d.phase)!!.let { if (it == "auto") "track" else it }, p.getString("batch", d.batch)!!, p.getString("replays", d.replays)!!,
                p.getString("focus", d.focus)!!)
        }
    }
}
