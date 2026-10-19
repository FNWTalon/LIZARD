package dev.lizard.receiver

import android.content.Context

// ai: The app's switches, in SharedPreferences "lizard"; Receive's Settings but replays, which is Developer Tools',
// ai: and precision, which only the tools set:
// ai:   decoder     auto | gpu | cpu                      ReceiverConfig.decoder
// ai:   precision   auto | int8 | f16 | f32               ReceiverConfig.precision; set only by tools/phone/ab.sh p= since
// ai:               2026-10-01 (its chips went)
// ai:   layout      1:1 | 2:1                            ReceiverConfig.layout (2:1: the frame's centre 2:1, cut in half)
// ai:   camera      a back camera's id (Engine.lenses); empty, or one the phone no longer lists, loads as the back camera
// ai:               that focuses closest (Engine.rearId) and is stored as that id at the next save (2026-10-05; "auto" a
// ai:               choice of its own until then, and a stored "auto" loads the same way)
// ai:   resolution  1280x720 | 1920x1080 | 2560x1440 | 3840x2160, the camera's ImageReader (1920x1080)
// ai:   zoom        the camera's zoom ratio, 0.1 apart over its range up to 4 (Settings' slider since 2026-10-01; chips
// ai:               of 1, 1.4 and 2 before); 1.5 by default since 2026-10-02 (the zoom of the 2:1 runs at 2.2+ MB/s;
// ai:               1.4 before, the 2026-09-30 sweep's best, 1.7 next): the code in the middle of the lens's field, not
// ai:               out to its soft corners (Engine.session; set live, Engine.zoom); a lens's in each crop (below)
// ai:   (focus      auto | dioptres, Receive's Focus of 2026-10-04, went 2026-10-05: autofocus, the camera's continuous
// ai:               video mode, always, Engine.applyFocus; a stored one is removed at the next save)
// ai:   (devlog     the rig's address for the stats rows, went 2026-10-09: rows are logged only in a replay; a stored
// ai:               one is removed at the next save)
// ai:   batch       1 to 32, the most frames a GPU launch takes (Receive's Settings, 2026-10-02; since 2026-10-08 a
// ai:               launch goes as soon as a frame waits and takes every frame then waiting up to this, where it
// ai:               waited for this many before); 32 by default (receiver.h batchCap); set live
// ai:   (phase      gone 2026-10-10: the camera's phase lock always runs track, PhaseLock.kt; it was off | track,
// ai:               track by default since 2026-10-01, and a stored value is removed at the next save. A sender that
// ai:               paints slower loses nothing to it. `debug.lizard.phase`, when set, overrides it for the tools.)
// ai:   replays     off | on: Save replays (Developer Tools, 2026-10-03; MainActivity's replay): the newest frames the
// ai:               decoder is handed kept while the camera runs; off by default, to spare storage. Since 2026-10-05 the
// ai:               switch is the session's alone, never saved (a recording left on costs every later run its lag):
// ai:               every start begins off, and a stored "on" from before is removed at the next save
// ai:   stats       on | off: Show statistics (Developer Tools, 2026-10-10): off hides Receive's green line once a file
// ai:               is in ("Received <name>, <size> in <s>, <speed>"; Receive.kt TransferPanel) and Developer Tools' lab
// ai:               line and readout (SettingsPanel.kt ReceiveAdvanced); off by default, kept
// ai: The camera's rate is no switch since 2026-10-01: [60,60], else the highest fixed range (Engine.pickFps).
// ai: Resolution is a lens's and zoom a lens's in each crop (2026-10-04; the crop too since 2026-10-18): resolution is
// ai: kept as "resolution@<camera id>" and zoom as "zoom@<camera id>/<layout>" for the back camera the camera switch
// ai: resolves to (Engine.rearId), and choosing another lens or crop brings its own back (MainActivity.change). Where
// ai: none is saved, zoom reads the lens's one zoom of before ("zoom@<camera id>"), then the plain key (the one value
// ai: every lens shared before 2026-10-04), else the default; resolution the plain key, else the default. A save writes
// ai: the lens's and crop's keys, gives the other crop the zoom it read until then where it has none of its own, and
// ai: drops the older keys. Where the phone has no back camera they stay under plain keys ("zoom/<layout>"). The other
// ai: switches are the app's, under their plain keys.
data class Settings(
    val decoder: String = "auto",
    val precision: String = "auto",
    val layout: String = "1:1",
    val camera: String = "",
    val resolution: String = "1920x1080",
    val zoom: String = "1.5",
    val batch: String = "32",
    val replays: String = "off",
    val stats: String = "off",
) {
    val frames get() = batch.toIntOrNull()?.coerceIn(1, 32) ?: 32
    // ai: cam: the id the camera switch resolves to (Engine.rearId), null where the phone has no back camera (the
    // ai: lens's values then stay under the plain keys)
    fun save(ctx: Context, cam: String?) {
        val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val e = p.edit()
            .putString("decoder", decoder).putString("precision", precision).putString("layout", layout).putString("camera", camera)
            .remove("fps").remove("devlog").remove("phase").putString("batch", batch).remove("replays")
            .putString("stats", stats)
        e.putString(key("resolution", cam), resolution); if (cam != null) e.remove("resolution")
        for (l in LAYOUTS) if (l != layout && !p.contains(zoomKey(cam, l))) e.putString(zoomKey(cam, l), zoomOf(p, cam, l))
        e.putString(zoomKey(cam, layout), zoom).remove(key("zoom", cam)).remove("zoom")
        e.remove("focus"); if (cam != null) e.remove(key("focus", cam))   // ai: the focus of 2026-10-04, stored until 2026-10-05
        e.apply()
    }

    // ai: this, with the lens's resolution and the lens's zoom in this crop as saved for cam (else the older keys', else
    // ai: the defaults)
    fun forCamera(ctx: Context, cam: String?): Settings {
        val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return copy(resolution = p.getString(key("resolution", cam), null) ?: p.getString("resolution", null) ?: Settings().resolution,
            zoom = zoomOf(p, cam, layout))
    }

    companion object {
        const val PREFS = "lizard"
        val DECODERS = listOf("auto", "gpu", "cpu")
        val PRECISIONS = listOf("auto", "int8", "f16", "f32")
        val LAYOUTS = listOf("1:1", "2:1")
        val RESOLUTIONS = listOf("1280x720", "1920x1080", "2560x1440", "3840x2160")

        private fun key(k: String, cam: String?) = if (cam == null) k else "$k@$cam"
        private fun zoomKey(cam: String?, layout: String) = "${key("zoom", cam)}/$layout"
        // ai: the zoom kept for cam in a crop: its own, else the lens's one zoom of before, else the plain one, else the default
        private fun zoomOf(p: android.content.SharedPreferences, cam: String?, layout: String) =
            p.getString(zoomKey(cam, layout), null) ?: p.getString(key("zoom", cam), null) ?: p.getString("zoom", null) ?: Settings().zoom

        // ai: rearId: the camera switch's value to the back camera it opens (Engine.rearId); the switch then holds that
        // ai: id (none where the phone has no back camera)
        fun load(ctx: Context, rearId: (String) -> String?): Settings {
            val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val d = Settings()
            val s = Settings(p.getString("decoder", d.decoder)!!, p.getString("precision", d.precision)!!,
                p.getString("layout", d.layout)!!, p.getString("camera", d.camera)!!, d.resolution, d.zoom,
                p.getString("batch", d.batch)!!, d.replays, p.getString("stats", d.stats)!!)
            val cam = rearId(s.camera)
            return s.copy(camera = cam ?: s.camera).forCamera(ctx, cam)
        }
    }
}
