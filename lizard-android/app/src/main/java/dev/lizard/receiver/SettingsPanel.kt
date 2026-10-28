package dev.lizard.receiver

import dev.lizard.readout.Readout
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import android.util.Range
import java.util.Locale
import kotlin.math.roundToInt
import org.json.JSONObject

// ai: Settings, as the web's pages hold them since 2026-10-02. Receive: under the transfer, Settings (the decoder and
// ai: its batch size, the camera: lens, resolution, format, zoom, crop), Developer Tools (Save replays, the lab line, the
// ai: readout: the decoder, camera, code, file, lock and heat a line each) and About,
// ai: the camera running beside them. Home: the tips again, the received files with Delete all, About. The Settings screen, reached by a gear on
// ai: Home and Receive, went with them. Options bare, no descriptions.

// ai: The decoder (2026-10-01: auto by default, the user free to switch; auto takes the GPU where the phone runs it,
// ai: else the C, Receiver::create): GPU and CPU alone since 2026-10-05, a menu since 2026-10-06 (an Auto chip naming
// ai: what auto last ran on, "Auto (GPU)", until 2026-10-05; "Decoding on the GPU." under the chips until 2026-10-02):
// ai: with nothing chosen the setting stays auto and the menu shows what auto runs on; a choice takes that decoder
// ai: outright. The lab's switches' menus save and restart as the chips did (MainActivity.change).
@Composable
internal fun MainActivity.ReceiveSettings() {
    val caps = remember(settings.camera) { runCatching { engine.caps(settings) }.getOrNull() }
    val s = settings
    Fields {
        // ai: the fields in the web's shape, a menu each (2026-10-06; chips until then); auto shows the decoder it runs
        // ai: PC only with a cable in (MainActivity.usb): the PC decoder is the phone's frames over USB to a PC
        val decoders = listOf("gpu" to "GPU", "cpu" to "CPU") + listOfNotNull(("pc" to "PC").takeIf { usb.connected || s.decoder == "pc" })
        Select("Decoder", if (s.decoder == "auto") autoRan.ifEmpty { "GPU" }.lowercase() else s.decoder, decoders) { change(s.copy(decoder = it), true) }
        // ai: the most frames a GPU launch takes (Settings.batch): a launch goes as soon as a frame waits, with every
        // ai: frame then waiting up to this; 1 decodes each frame alone. The C decodes a frame at a time: not shown with it.
        if (s.decoder == "pc") PcLinkField()
        if (s.decoder != "cpu" && s.decoder != "pc") Field("Batch size", "${s.frames}") {
            Bar(s.frames.toFloat(), 1f..32f, 30) { v -> val n = v.roundToInt().coerceIn(1, 32); if (n != s.frames) change(s.copy(batch = n.toString()), false) }
        }
        // ai: the back cameras by lens, one chosen (2026-10-05; an Auto chip, the closest-focusing lens, until then: the
        // ai: setting now holds that lens's id itself, Settings.load)
        Select("Camera", s.camera, caps?.lenses.orEmpty().map { it.id to it.label }, blank = "Camera") { change(s.copy(camera = it), true) }
        Select("Resolution", s.resolution, Settings.RESOLUTIONS.map { it to it }, enabledFor = { caps == null || it in caps.sizes }) { change(s.copy(resolution = it), true) }
        // ai: the camera's format (Settings.format, 2026-10-27, as the desktop's Format): Auto, or each the lens offers at
        // ai: this resolution (RAW at any: the lens's whole array) that the decoder reads (Private with the GPU decoder
        // ai: alone); shown where there is a choice
        val fmts = Settings.FORMATS.filter { (k, _) ->
            val z = caps?.formats?.get(k)
            z != null && (k.startsWith("raw") || s.resolution in z) && !(k == "private" && (s.decoder == "cpu" || s.decoder == "pc"))
        }
        if (fmts.size > 1 || s.format != "auto") Select("Format", s.format, listOf("auto" to "Auto") + fmts) { change(s.copy(format = it), true) }
        ZoomField(caps?.zoom)
        Select("Crop", s.layout, Settings.LAYOUTS.map { it to it }) { change(s.copy(layout = it), true) }
    }
}

// ai: The PC decoder's link (PcLink, the stream's stats): connected, the PC's address ("USB" for the accessory) and the
// ai: frames a second it carries; else "disconnected", the USB accessory where it is open and the USB tether's address
// ai: (no Wi-Fi: the PC decoder is the cable's, 2026-10-27), and the tethering page where the cable has neither. Read at each stats poll (a second).
@Composable
internal fun MainActivity.PcLinkField() {
    val st = remember(raw) { runCatching { JSONObject(raw).optJSONObject("stream") }.getOrNull() }
    val addrs = remember(raw, usb) { PcLink.usbAddresses() }
    val accessory = remember(raw, usb) { Native.accessoryOpen() }
    val port = st?.optInt("port", 47230) ?: 47230
    val connected = st?.optBoolean("connected") == true
    Field("PC link", if (connected) "to ${st?.optString("peer").orEmpty()}" else "disconnected") {
        val lines = if (connected) listOf(String.format(Locale.ROOT, "%.0f of %.0f frames a second", st?.optDouble("sentFps", 0.0) ?: 0.0, st?.optDouble("cameraFps", 0.0) ?: 0.0))
            else (listOfNotNull("USB accessory".takeIf { accessory }) + addrs.map { "USB $it:$port" }).ifEmpty { listOf("USB tethering off") }
        for (l in lines) Text(l, style = MaterialTheme.typography.bodyMedium, color = Muted)
        if (usb.connected && !usb.tether && !accessory && !connected && addrs.isEmpty())
            Btn("USB tethering", modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) { PcLink.openTethering(this@PcLinkField) }
    }
}

// ai: Experimental (below Settings): a lab arm a user may try. D6 phase lock (Settings.d6): the rework's arm D6 in place
// ai: of the phase lock, taken by the next camera session (the camera restarts)
@Composable
internal fun MainActivity.ReceiveExperimental() {
    val s = settings
    Fields {
        Select("D6 phase lock", s.d6, listOf("off" to "Off", "on" to "On")) { change(s.copy(d6 = it), true) }
    }
}

// ai: The lab's readouts: Save replays (MainActivity's runs: the switch, each run listed with its Download once ended,
// ai: how the last Download went), the lab line, then the readout (devReadout).
@Composable
internal fun MainActivity.ReceiveAdvanced() {
    val s = settings
    // ai: one rhythm, Settings' (Fields: 16 dp between every part, 2026-10-10; the two menus touched and the readouts and
    // ai: Download carried 6 dp of their own until then); Show statistics first, then Save replays and its runs
    Fields {
        Select("Show statistics", s.stats, listOf("off" to "Off", "on" to "On")) { change(s.copy(stats = it), false) }
        // ai: none for the PC decoder: the stream receiver keeps no frames, the PC's own Save replays records them
        if (s.decoder != "pc") Select("Save replays", s.replays, listOf("off" to "Off", "on" to "On")) { change(s.copy(replays = it), false) }
        // ai: the runs in the order begun: the newest finished, then those still recording or saving
        for (rp in MainActivity.runs) {
            if (s.stats == "on") CodeBlock(when (rp.state) {
                MainActivity.ReplayState.Recording -> "${rp.run}: recording"
                MainActivity.ReplayState.Ending -> "${rp.run}: saving"
                MainActivity.ReplayState.Ready -> "${rp.run}: ${rp.frames} frames, ${"%.0f".format(Locale.ROOT, rp.bytes / 1e6)} MB" +
                    if (rp.why.isEmpty()) "" else " (${rp.why})"
                MainActivity.ReplayState.Failed -> "${rp.run}: ${rp.why}"
            }, size = 12)
            if (rp.state == MainActivity.ReplayState.Ready)
                Btn(if (MainActivity.exporting == rp.run) "Downloading" else "Download", enabled = MainActivity.exporting.isEmpty(),
                    modifier = Modifier.fillMaxWidth()) { downloadReplay(rp) }
        }
        if (s.stats == "on" && MainActivity.replayNote.isNotEmpty()) CodeBlock(MainActivity.replayNote, size = 12)
        // ai: the statistics (the lab line and the readout), like every monospace line here (a replay's line, the note after a
        // ai: download), only under Show statistics (2026-10-10), as Receive's green line; the switches and Download stay
        if (s.stats == "on") {
            val lab = Readout.lab(phase == Engine.Phase.On, rx)
            if (lab.isNotEmpty()) CodeBlock(lab)
            CodeBlock(devReadout(), size = 12)
        }
    }
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
                "batches up to ${j.optInt("cap")}, ${f(j.optDouble("gpuMs", 0.0), 2)} ms of GPU a frame"
            else if (dec == "cpu") "CPU, ${j.optInt("threads")} of ${j.optInt("threadsMax")} threads${if (j.optBoolean("simd")) ", NEON" else ""}, " +
                "${f(j.optDouble("cpuMs", 0.0), 1)} ms a frame"
            // ai: PC (liblizard/core/rx/stream.h): the stream's own line, the PC's decode after it
            else if (dec.startsWith("pc")) j.optJSONObject("stream")?.let { st ->
                (if (st.optBoolean("connected")) "to the PC at ${st.optString("peer")}" else "waiting for the PC on port ${st.optInt("port")}") +
                    ", ${f(st.optDouble("sentFps", 0.0), 0)} of ${f(st.optDouble("cameraFps", 0.0), 0)} frames a second sent, ${f(st.optDouble("MBps", 0.0), 1)} MB/s, " +
                    "${st.optLong("dropped")} dropped${dec.removePrefix("pc").removePrefix(":").trim().let { if (it.isEmpty()) "" else "; the PC's $it" }}"
            } ?: dec
            else dec,
            j.optString("gpuWhy").takeIf { dec == "cpu" && it.isNotEmpty() }?.let { "no GPU: $it" })
    }
    val c = cam
    val now = engine.cameraNow()
    put("camera", c?.let { "${it.id}: ${it.size.width}x${it.size.height} ${it.format} at ${it.fps.upper} fps, ${settings.layout} crop${if (it.note.isEmpty()) "" else ", ${it.note}"}" } ?: "not open",
        j?.let { "${f(it.optDouble("capturedFps", 0.0), 0)} captured, ${f(it.optDouble("processedFps", 0.0), 0)} decoded, ${it.optInt("dropped")} dropped a second" +
            if (it.has("heldMs")) ", held ${f(it.optDouble("heldMs", 0.0), 0)} ms (${f(it.optDouble("heldMaxMs", 0.0), 0)} max)" else "" },
        now?.let { "exposure ${f(it.exposureMs, 2)} ms, ISO ${it.iso}, readout ${f(it.readoutMs, 2)} ms" + (it.lens?.let { l -> ", lens $l" } ?: "") })
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
    } ?: engine.lockBar?.let { put("lock", "off: $it") }
    put("heat", "thermal $heat" + (clocks?.let { ", GPU ${it.mhz} of ${it.top} MHz" } ?: "") + (powerW?.let { ", ${f(it, 1)} W" } ?: "") +
        (if (batteryC > 0) ", battery ${f(batteryC, 1)} C" else ""))
    j?.optString("error")?.takeIf { it.isNotEmpty() }?.let { put("error", it) }
    return out.joinToString("\n")
}

// ai: Zoom as a slider (2026-10-01; chips of 1, 1.4 and 2 before):
// ai: 0.1x steps over the camera's whole range (up to 4x until 2026-10-21: the S26's camera 0 runs 0.6 to 10x and reaches
// ai: its 3x and 5x lenses by the zoom, Engine.CameraNow's lens) (on the S26's 2.2 mm camera, aimed by hand to fill the square,
// ai: 1.0x read 381 to 420 KB/s, 1.2x 456, 1.4x 509 and 589, 1.7x 517, 2.8x 352: STATUS "The Android app live on
// ai: the S26"), moved live on the running camera while it slides (Engine.zoom) and kept at every step (a press that a
// ai: scroll then took, on the S26, moved it with no "finished" call, so a value kept only then could differ from the label).
@Composable
private fun MainActivity.ZoomField(range: Range<Float>?) {
    val lo = range?.lower ?: 1f
    val hi = maxOf(lo + 0.1f, range?.upper ?: 4f)
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
