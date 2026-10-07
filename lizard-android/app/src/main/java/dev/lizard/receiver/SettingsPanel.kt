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
// ai: the transfer, Settings (the decoder and its frames a batch, the camera: lens, resolution, zoom, crop, phase lock), Advanced (the lab
// ai: line, the dev log's address, the camera's and the receiver's raw readout) and About, the camera running beside
// ai: them. Home: the tips again, the received files with Delete all, About. The Settings screen, reached by a gear on
// ai: Home and Receive, went with them. Options bare, no descriptions.

// ai: The decoder (2026-10-01: auto by default, the user free to switch; auto takes the GPU where the phone runs it,
// ai: else the C, Receiver::create): chips GPU and CPU alone since 2026-10-05 (an Auto chip naming what auto last ran
// ai: on, "Auto (GPU)", until then; "Decoding on the GPU." under the chips until 2026-10-02): with nothing chosen the
// ai: setting stays auto and the chip of what auto runs on is marked; a tap chooses that decoder outright. The lab's
// ai: switches (their chips save and restart as they did, MainActivity.change; their keys unchanged, tools/phone/ab.sh
// ai: rewrites them).
@Composable
internal fun MainActivity.ReceiveSettings() {
    val caps = remember(settings.camera) { runCatching { engine.caps(settings) }.getOrNull() }
    val s = settings
    Fields {
        // ai: the fields in the web's shape, a menu each (2026-10-06; chips until then); auto shows the decoder it runs
        Select("Decoder", if (s.decoder == "auto") autoRan.ifEmpty { "GPU" }.lowercase() else s.decoder, listOf("gpu" to "GPU", "cpu" to "CPU")) { change(s.copy(decoder = it), true) }
        // ai: the most frames a GPU batch waits for (Settings.batch): 1 hands each capture's reading to the phase lock
        // ai: as it is decoded, for a screen whose cadence slips; 32 the least GPU work a frame (the S26: 6.58 ms a
        // ai: frame at 1, 2.88 at 32). The C decodes a frame at a time: not shown with it.
        if (s.decoder != "cpu") Field("Batch size", "${s.frames}") {
            Bar(s.frames.toFloat(), 1f..32f, 30) { v -> val n = v.roundToInt().coerceIn(1, 32); if (n != s.frames) change(s.copy(batch = n.toString()), false) }
        }
        // ai: the back cameras by lens, one chosen (2026-10-05; an Auto chip, the closest-focusing lens, until then: the
        // ai: setting now holds that lens's id itself, Settings.load)
        Select("Camera", s.camera, caps?.lenses.orEmpty().map { it.id to it.label }, blank = "Camera") { change(s.copy(camera = it), true) }
        Select("Resolution", s.resolution, Settings.RESOLUTIONS.map { it to it }, enabledFor = { caps == null || it in caps.sizes }) { change(s.copy(resolution = it), true) }
        ZoomField(caps?.zoom)
        Select("Crop", s.layout, Settings.LAYOUTS.map { it to if (it == "2:1") "2:1 (experimental)" else it }) { change(s.copy(layout = it), true) }
        Select("Phase lock", s.phase, Settings.PHASES.map { it to it.replaceFirstChar { c -> c.uppercase() } }) { change(s.copy(phase = it), false) }
    }
}

// ai: The lab's readouts: Save replays (MainActivity's runs: the switch, each run listed with its Download once ended,
// ai: how the last Download went), the lab line, the dev log's address, then the
// ai: camera's and the receiver's raw readout.
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
    Spacer(Modifier.height(16.dp))
    TextInput("Dev log address", s.devlog, "http://host:8080") { change(s.copy(devlog = it.trim()), false) }
    Spacer(Modifier.height(16.dp))
    val c = cam
    val camLine = if (c == null) "camera: not open" else
        "camera ${c.id}: ${c.format} ${c.size.width}x${c.size.height}, preview ${c.preview.width}x${c.preview.height}, " +
            "fps [${c.fps.lower},${c.fps.upper}], min frame %.2f ms, sensor %d°%s".format(c.minFrameMs, c.sensorOrientation,
                if (c.note.isEmpty()) "" else ", ${c.note}")
    val forced = if (s.precision != "auto") "\nprecision ${s.precision} (set by tools/phone/ab.sh p=)" else ""
    val pretty = runCatching { JSONObject(raw).toString(1) }.getOrDefault(raw)
    CodeBlock("$camLine$forced\n$pretty", size = 12)
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
