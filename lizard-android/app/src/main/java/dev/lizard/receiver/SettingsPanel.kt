package dev.lizard.receiver

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import android.util.Range
import java.util.Locale
import kotlin.math.roundToInt
import org.json.JSONObject

// ai: Settings, as the web's pages hold them since 2026-10-02. Receive: under
// ai: the transfer, Settings (the decoder and its batch size, the camera: lens, resolution, zoom, crop, phase lock), Developer Tools (Save
// ai: replays, the lab line, the readout: the decoder, camera, code, file, lock and heat a line each) and About, the camera running beside
// ai: them. Home: the tips again, the received files with Delete all, About. The Settings screen, reached by a gear on
// ai: Home and Receive, went with them. Options bare, no descriptions.

// ai: The decoder (2026-10-01: auto by default, the user free to switch; auto takes the GPU where the phone runs it,
// ai: else the C, Receiver::create): GPU and CPU alone since 2026-10-05, a menu since 2026-10-06 (an Auto chip naming
// ai: what auto last ran on, "Auto (GPU)", until 2026-10-05; "Decoding on the GPU." under the chips until 2026-10-02):
// ai: with nothing chosen the setting stays auto and the menu shows what auto runs on; a choice takes that decoder
// ai: outright. The lab's switches (their menus save and restart as the chips did, MainActivity.change; their keys
// ai: unchanged, tools/phone/ab.sh rewrites them).
@Composable
internal fun MainActivity.ReceiveSettings() {
    val caps = remember(settings.camera) { runCatching { engine.caps(settings) }.getOrNull() }
    val s = settings
    Fields {
        // ai: the fields in the web's shape, a menu each (2026-10-06; chips until then); auto shows the decoder it runs
        Select("Decoder", if (s.decoder == "auto") autoRan.ifEmpty { "GPU" }.lowercase() else s.decoder, listOf("gpu" to "GPU", "cpu" to "CPU")) { change(s.copy(decoder = it), true) }
        // ai: the most frames a GPU launch takes (Settings.batch): a launch goes as soon as a frame waits, with every
        // ai: frame then waiting up to this; 1 decodes each frame alone. The C decodes a frame at a time: not shown with it.
        if (s.decoder != "cpu") Field("Batch size", "${s.frames}") {
            Bar(s.frames.toFloat(), 1f..32f, 30) { v -> val n = v.roundToInt().coerceIn(1, 32); if (n != s.frames) change(s.copy(batch = n.toString()), false) }
        }
        // ai: the back cameras by lens, one chosen (2026-10-05; an Auto chip, the closest-focusing lens, until then: the
        // ai: setting now holds that lens's id itself, Settings.load)
        Select("Camera", s.camera, caps?.lenses.orEmpty().map { it.id to it.label }, blank = "Camera") { change(s.copy(camera = it), true) }
        Select("Resolution", s.resolution, Settings.RESOLUTIONS.map { it to it }, enabledFor = { caps == null || it in caps.sizes }) { change(s.copy(resolution = it), true) }
        ZoomField(caps?.zoom)
        Select("Crop", s.layout, Settings.LAYOUTS.map { it to if (it == "2:1") "2:1 (experimental)" else it }) { change(s.copy(layout = it), true) }
    }
}

// ai: The lab's readouts: Save replays (MainActivity's runs: the switch, each run listed with its Download once ended,
// ai: how the last Download went), the lab line, then the readout (devReadout). The dev log's address and the stats
// ai: JSON printed whole went 2026-10-09 (rows are logged only in a replay; the JSON's 150-frame series made it a page).
@Composable
internal fun MainActivity.ReceiveAdvanced() {
    val s = settings
    Select("Save replays", s.replays, listOf("off" to "Off", "on" to "On")) { change(s.copy(replays = it), false) }
    // ai: the runs in the order begun: the newest finished, then those still recording or saving
    for (rp in MainActivity.runs) {
        CodeBlock(when (rp.state) {
            MainActivity.ReplayState.Recording -> "${rp.run}: recording"
            MainActivity.ReplayState.Ending -> "${rp.run}: saving"
            MainActivity.ReplayState.Ready -> "${rp.run}: ${rp.frames} frames, ${"%.0f".format(Locale.ROOT, rp.bytes / 1e6)} MB" +
                if (rp.why.isEmpty()) "" else " (${rp.why})"
            MainActivity.ReplayState.Failed -> "${rp.run}: ${rp.why}"
        }, size = 12)
        if (rp.state == MainActivity.ReplayState.Ready)
            Btn(if (MainActivity.exporting == rp.run) "Downloading" else "Download", enabled = MainActivity.exporting.isEmpty(),
                modifier = Modifier.fillMaxWidth().padding(top = 6.dp, bottom = 6.dp)) { downloadReplay(rp) }
    }
    if (MainActivity.replayNote.isNotEmpty()) CodeBlock(MainActivity.replayNote, size = 12)
    val lab = Readout.lab(phase == Engine.Phase.On, rx)
    if (lab.isNotEmpty()) CodeBlock(lab)
    CodeBlock(devReadout(), size = 12)
}

// ai: The readout, a labelled line or two a part, from the receiver's stats JSON (receiver.cpp stats), the camera
// ai: (Engine.CamInfo, cameraNow), the phase lock's snapshot and the heat; a part with nothing to say is left out.
private fun MainActivity.devReadout(): String {
    val j = runCatching { JSONObject(raw) }.getOrNull()
    val f = { x: Double, d: Int -> String.format(Locale.ROOT, "%.${d}f", x) }
    val out = mutableListOf<String>()
    fun put(label: String, vararg lines: String?) {
        lines.filterNotNull().filter { it.isNotEmpty() }.forEachIndexed { i, l -> out.add((if (i == 0) label else "").padEnd(9) + l) }
    }
    if (j != null) {
        val dec = j.optString("decoder")
        put("decoder", if (dec.startsWith("gpu")) "GPU ${dec.removePrefix("gpu").trim()}${if (j.optBoolean("zeroCopy")) ", zero-copy" else ""}, " +
                "batches up to ${j.optInt("cap", j.optInt("B"))}, ${f(j.optDouble("gpuMs", 0.0), 2)} ms of GPU a frame"
            else if (dec == "cpu") "CPU, ${j.optInt("threads")} of ${j.optInt("threadsMax")} threads${if (j.optBoolean("simd")) ", NEON" else ""}, " +
                "${f(j.optDouble("cpuMs", 0.0), 1)} ms a frame" else dec,
            j.optString("gpuWhy").takeIf { dec == "cpu" && it.isNotEmpty() }?.let { "no GPU: $it" })
    }
    val c = cam
    val now = engine.cameraNow()
    put("camera", c?.let { "${it.id}: ${it.size.width}x${it.size.height} ${it.format} at ${it.fps.upper} fps, ${settings.layout} crop${if (it.note.isEmpty()) "" else ", ${it.note}"}" } ?: "not open",
        j?.let { "${f(it.optDouble("capturedFps", 0.0), 0)} captured, ${f(it.optDouble("processedFps", 0.0), 0)} decoded, ${it.optInt("dropped")} dropped a second" +
            if (it.has("heldMs")) ", held ${f(it.optDouble("heldMs", 0.0), 0)} ms (${f(it.optDouble("heldMaxMs", 0.0), 0)} max)" else "" },
        now?.let { "exposure ${f(it.exposureMs, 2)} ms, ISO ${it.iso}, readout ${f(it.readoutMs, 2)} ms" })
    if (j != null) {
        val word = j.optJSONObject("word")
        val v = word?.optInt("version", 0) ?: 0
        put("code", "found ${f(100 * j.optDouble("foundShare", 0.0), 0)}%" + (if (j.optDouble("side", 0.0) > 0) ", symbol ${f(j.optDouble("side"), 0)} px" else "") +
                (if (v > 0) ", LIZARD-${8 * v}${word?.optInt("fps", 0)?.takeIf { it > 0 }?.let { " at $it fps" } ?: ""}" else ", no word yet"),
            "${Readout.rate(j.optDouble("goodputKBs", 0.0))}, ${j.optInt("blocks")} blocks in ${f(j.optDouble("windowSecs", 0.0), 2)} s" +
                (j.optJSONObject("totals")?.let { t -> "; ${t.optLong("blocks")} blocks in ${t.optLong("frames")} frames all told" } ?: ""))
        val file = j.optJSONObject("file")
        put("file", when {
            j.optBoolean("test") -> "the test stream"
            file == null -> "none yet"
            else -> "${file.optString("name").ifEmpty { "(no name)" }}: ${Readout.partOf(file.optLong("received").toDouble(), file.optLong("size"))}" +
                ", chunks ${file.optInt("chunksVerified")} of ${file.optInt("chunks")} verified${if (file.optBoolean("verified")) ", whole" else ""}"
        })
    }
    engine.phaseState()?.let { p ->
        val pct = { x: Double -> if (x.isNaN()) "-" else "${f(100 * x, 0)}%" }
        put("lock", "${p.arm} ${p.what}: ${pct(p.a)} before, ${pct(p.b)} after; gain ${f(p.gain, 2)}, pace ${f(p.pace, 0)} us a second, ${p.delays} delays")
    }
    put("heat", "thermal $heat" + (clocks?.let { ", GPU ${it.mhz} of ${it.top} MHz" } ?: "") + (powerW?.let { ", ${f(it, 1)} W" } ?: "") +
        (if (batteryC > 0) ", battery ${f(batteryC, 1)} C" else ""))
    if (settings.precision != "auto") put("precision", "${settings.precision} (set by tools/phone/ab.sh p=)")
    j?.optString("error")?.takeIf { it.isNotEmpty() }?.let { put("error", it) }
    return out.joinToString("\n")
}

// ai: Zoom as a slider (2026-10-01; chips of 1, 1.4 and 2 before):
// ai: 0.1x steps over the camera's range up to 4x (on the S26's 2.2 mm camera, aimed by hand to fill the square,
// ai: 1.0x read 381 to 420 KB/s, 1.2x 456, 1.4x 509 and 589, 1.7x 517, 2.8x 352: STATUS "The Android app live on
// ai: the S26"), moved live on the running camera while it slides (Engine.zoom) and kept at every step (a press that a
// ai: scroll then took, on the S26, moved it with no "finished" call, so a value kept only then could differ from the label).
@Composable
private fun MainActivity.ZoomField(range: Range<Float>?) {
    val lo = range?.lower ?: 1f
    val hi = maxOf(lo + 0.1f, minOf(range?.upper ?: 4f, 4f))
    val z = (settings.zoom.toFloatOrNull() ?: 1.5f).coerceIn(lo, hi)
    Field("Zoom", "%.1fx".format(Locale.ROOT, z)) {
        Bar(z, lo..hi, maxOf(0, ((hi - lo) * 10).roundToInt() - 1)) { zoomTo((it * 10).roundToInt() / 10f) }
    }
}

// ai: Home's settings rows (the web Home's #settings): the tips again, the received files' count and size with Delete
// ai: all (asked first), About.
@Composable
internal fun MainActivity.HomeSettings() {
    var asking by remember { mutableStateOf(false) }
    SectionLabel("Settings")
    ListRow("Show tips again", "The first-run tips, on every page", onClick = { tipsAgain() })
    val total = files.sumOf { it.size }
    ListRow("Received files", if (files.isEmpty()) "No files kept" else "${files.size} file${if (files.size > 1) "s" else ""} kept, ${Readout.bytes(total)}") {
        if (files.isNotEmpty()) Btn("Delete all", Kind.Danger) { asking = true }
    }
    ListRow("About", aboutLine(), divider = false)
    if (asking) AlertDialog(onDismissRequest = { asking = false },
        confirmButton = { Btn("Delete all", Kind.Danger) { asking = false; deleteAll() } },
        dismissButton = { Btn("Cancel", Kind.Text) { asking = false } },
        title = { Text("Delete ${files.size} file${if (files.size > 1) "s" else ""}?") },
        text = { Text("They go from this phone. Copies saved elsewhere stay.") },
        containerColor = Bg)
}

// ai: About, a tool screen's last row (the web's .item.about), a hairline over it
@Composable
internal fun MainActivity.About() = Column(Modifier.fillMaxWidth()) {
    HorizontalDivider(color = Line)
    ListRow("About", aboutLine(), divider = false)
}

internal fun MainActivity.aboutLine() = "LIZARD ${version()}${installed().let { if (it.isEmpty()) "" else ", installed $it" }}"
