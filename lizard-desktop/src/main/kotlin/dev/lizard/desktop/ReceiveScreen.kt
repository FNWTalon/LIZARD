package dev.lizard.desktop

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.lizard.readout.Readout
import java.util.Locale
import kotlin.math.ln
import kotlin.math.exp
import kotlin.math.roundToInt

// ai: Receive (2026-10-25 on the desktop; the web's recv.html in landscape, the Android app's Receive.kt): a 320 dp
// ai: column on the left, the bar (Receive, the swap to the sender, the collapser) over the rest, which scrolls on its
// ai: own: the actions (once a file is in Open, the solid one, and Show in folder, which give way while the test stream
// ai: is read; then Start camera or Stop camera), the state line, the meter, the figures, one hint where it applies
// ai: (muted, under the figures), Settings, Developer Tools, About; a hairline; the camera's crop on the right, as big
// ai: as the rest of the window allows, centred. Collapsed, the column is the 56 dp rail: play or pause, the rate (while
// ai: live) and the percent while a file or the test stream is read, a green check and Open once the file is in, Home
// ai: at its foot. The words are Readout.kt's.
@Composable
fun ReceiveScreen(app: App) {
    val r = app.recv
    Row(Modifier.fillMaxSize().background(Bg)) {
        if (r.collapsed) Rail(onClick = { r.toggleRail() }, foot = { Square { IconBtn(Icons.back, "Home") { app.go(Screen.Home) } } }) { RailSquares(r) }
        else Column(Modifier.width(320.dp).fillMaxHeight()) {
            Box(Modifier.padding(horizontal = 16.dp)) { TopBar("Receive") { SwapBtn("Sender") { app.go(Screen.Send) }; CollapseBtn(false) { r.toggleRail() } } }
            Box(Modifier.weight(1f).fillMaxWidth()) {
                val scroll = rememberScrollState()
                Column(Modifier.fillMaxSize().verticalScroll(scroll).padding(horizontal = 16.dp)) { Side(app, r) }
                VerticalScrollbar(rememberScrollbarAdapter(scroll), Modifier.align(Alignment.CenterEnd).fillMaxHeight())
            }
        }
        VerticalDivider(color = Line)
        BoxWithConstraints(Modifier.weight(1f).fillMaxHeight().padding(16.dp), contentAlignment = Alignment.Center) {
            val ratio = r.cropRatio
            val w = minOf(maxHeight * ratio, maxWidth)
            Preview(r, Modifier.width(w).height(w / ratio))
        }
    }
}

private fun toneColour(t: Readout.Tone) = when (t) { Readout.Tone.Bad -> Bad; Readout.Tone.Good -> Good; Readout.Tone.Plain -> Fg }

// ai: The rail's squares (the web's #rail, the app's RailSquares): the rate only while the figures are this stream's
// ai: (ReceiveState.live), none with the camera off
@Composable
private fun RailSquares(r: ReceiveState) {
    val reading = r.on && ((r.rx.hasFile && !r.rx.verified) || r.testing)
    val kept = r.kept
    Square {
        IconBtn(if (r.running) Icons.pause else Icons.play, if (r.running) "Stop camera" else "Start camera",
            enabled = r.phase != ReceiveState.Phase.Starting) { r.toggle() }
    }
    if (kept == null) {
        RateSquare(if (reading && r.live) r.rx.goodputKBs else null)
        if (reading && !r.testing) PctSquare(Readout.fraction(r.rx))
    } else {
        Square { Icon(Icons.check, contentDescription = "Received ${kept.name}", Modifier.size(24.dp), tint = Good) }
        Square { IconBtn(Icons.open, "Open ${kept.name}") { Received.open(kept.file) } }
    }
}

@Composable
private fun ColumnScope.Side(app: App, r: ReceiveState) {
    val kept = r.kept
    Spacer(Modifier.height(12.dp))
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (kept != null) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Btn("Open", Kind.Primary, modifier = Modifier.weight(1f)) { Received.open(kept.file) }
            Btn("Show in folder", pad = 16.dp) { Received.show(kept.file) }
        }
        // ai: one solid button at a time: Start camera while nothing waits, else Open
        Btn(if (r.running) "Stop camera" else "Start camera", if (!r.running && kept == null) Kind.Primary else Kind.Tonal,
            enabled = r.phase != ReceiveState.Phase.Starting, modifier = Modifier.fillMaxWidth()) { r.toggle() }
    }
    val ln = r.line
    // ai: Show statistics off (Developer Tools, the app's 2026-10-10): no green line once a file is in, the meter and the
    // ai: buttons as before
    val shown = ln.text.isNotEmpty() && !(ln.tone == Readout.Tone.Good && !r.showStats)
    if (shown && ln.name != null) NameLine(ln.before, ln.name, ln.after, MaterialTheme.typography.titleMedium, toneColour(ln.tone), Modifier.padding(top = 12.dp, bottom = 2.dp))
    else if (shown) Text(ln.text, style = MaterialTheme.typography.titleMedium, color = toneColour(ln.tone), modifier = Modifier.padding(top = 12.dp, bottom = 2.dp))
    Meter(ln.frac)
    if (ln.nums.isNotEmpty()) Text(ln.nums, style = MaterialTheme.typography.bodyMedium, color = Muted)
    // ai: why the file is not in Downloads (ReceiveState.fileNote: it opens from the app's own folder meanwhile), in the
    // ai: error's colour, until a later filing puts it there
    r.fileNote.let { if (it.isNotEmpty()) Text(it, style = MaterialTheme.typography.bodyMedium, color = Bad, modifier = Modifier.padding(top = 4.dp)) }
    // ai: the hint (ReceiveState.hint: one, the first that holds) under the figures, as they are: muted
    r.hint.let { if (it.isNotEmpty()) Text(it, style = MaterialTheme.typography.bodyMedium, color = Muted, modifier = Modifier.padding(top = 4.dp)) }
    Spacer(Modifier.height(16.dp))
    Fold("Settings", r.settingsOpen, { r.toggleSettings() }) { ReceiveSettings(r) }
    Fold("Developer Tools", r.toolsOpen, { r.toggleTools() }) { ReceiveTools(r) }
    HorizontalDivider(color = Line)
    ListRow("About", app.about.ifEmpty { null })
    Spacer(Modifier.height(24.dp))
}

// ai: Settings (the web's #dev, the app's ReceiveSettings): a menu or a slider each, the camera's controls where it has them
@Composable
private fun ReceiveSettings(r: ReceiveState) {
    val ci = r.info
    val m = r.mode
    Fields {
        // ai: GPU and CPU; with neither chosen the decoder is auto's and the one it runs on is shown
        Select("Decoder", if (r.decoder == "auto") r.autoRan.ifEmpty { "GPU" }.lowercase() else r.decoder, listOf("gpu" to "GPU", "cpu" to "CPU")) { r.chooseDecoder(it) }
        Select("Camera", r.camera.ifEmpty { r.cameras.firstOrNull()?.id ?: "" }, r.cameras.map { it.id to it.label }, blank = "Camera",
            onOpen = { r.refreshCameras() }) { r.chooseCamera(it) }
        // ai: the sizes and rates the camera offers, once one has been opened (none to offer before)
        if (ci != null && m != null) {
            val sizes = ci.modes.distinctBy { it.size }.sortedBy { it.w.toLong() * it.h }
            Select("Resolution", m.size, sizes.map { it.size to "${it.w} x ${it.h}" }) { r.chooseRes(it) }
            val rates = ci.modes.filter { it.size == m.size }.map { it.fps }.distinctBy { ReceiveState.fpsText(it) }.sorted()
            Select("Frame rate", ReceiveState.fpsText(m.fps), rates.map { ReceiveState.fpsText(it) to ReceiveState.fpsText(it) }) { v ->
                rates.firstOrNull { ReceiveState.fpsText(it) == v }?.let { r.chooseFps(it) }
            }
        }
        Select("Crop", r.layout, listOf("1:1" to "1:1", "2:1" to "2:1 (experimental)")) { r.chooseLayout(it) }
        ci?.control("zoom")?.let { c -> ZoomField(r, c) }
        ci?.control("exposure")?.let { c -> ExposureField(r, c) }
        ci?.control("gain")?.let { c -> GainField(r, c) }
        // ai: the most frames a GPU launch takes (the app's Batch size): none for the C, which decodes a frame at a time
        if (!r.cpu) Field("Batch size", "${r.batch}") {
            Bar(r.batch.toFloat(), 1f..32f, 30, done = { r.commitBatch() }) { r.chooseBatch(it.roundToInt()) }
        }
    }
}

// ai: a slider's steps over a control's range: its own step where that makes 200 or fewer, else a continuous slider
private fun stepsOf(c: ReceiveState.Control): Int {
    if (c.step <= 0) return 0
    val n = ((c.max - c.min) / c.step).roundToInt() - 1
    return if (n in 0..200) n else 0
}
private fun snap(c: ReceiveState.Control, v: Double): Double {
    val x = if (c.step > 0) c.min + ((v - c.min) / c.step).roundToInt() * c.step else v
    return x.coerceIn(c.min, c.max)
}

// ai: Zoom (UVC zoom_absolute), live: shown as a ratio of its least ("1.5x") where it starts above 0
@Composable
private fun ZoomField(r: ReceiveState, c: ReceiveState.Control) {
    val z = (r.zoom ?: c.value).coerceIn(c.min, c.max)
    val label = when {
        c.unit == "x" -> String.format(Locale.ROOT, "%.1fx", z)
        c.min > 0 -> String.format(Locale.ROOT, "%.1fx", z / c.min)
        else -> String.format(Locale.ROOT, "%.0f", z)
    }
    Field("Zoom", label) { Bar(z.toFloat(), c.min.toFloat()..c.max.toFloat(), stepsOf(c)) { r.zoomTo(snap(c, it.toDouble())) } }
}

// ai: Exposure in us on a log scale (a camera's range spans decades: 10 us to seconds), live; its first step the
// ai: camera's own (Auto) where the camera can expose by itself, which is the default
private const val EXPOSURE_STEPS = 100
@Composable
private fun ExposureField(r: ReceiveState, c: ReceiveState.Control) {
    val lo = maxOf(c.min, 1.0); val hi = maxOf(c.max, lo + 1)
    val first = if (c.canAuto) 1 else 0
    val n = EXPOSURE_STEPS
    fun usAt(x: Int) = snap(c, lo * exp(ln(hi / lo) * (x - first).toDouble() / (n - first)))
    fun xOf(us: Double) = first + ((ln(us.coerceIn(lo, hi) / lo) / ln(hi / lo)) * (n - first)).roundToInt()
    val us = r.exposure
    val x = if (us == null) (if (c.canAuto) 0 else xOf(c.value)) else xOf(us)
    val label = if (us == null && c.canAuto) "Auto" else usText(us ?: c.value)
    Field("Exposure", label) {
        Bar(x.toFloat(), 0f..n.toFloat(), n - 1) { v ->
            val k = v.roundToInt()
            if (k < first) r.exposureTo(null) else r.exposureTo(usAt(k))
        }
    }
}
private fun usText(us: Double) = when {
    us < 1000 -> "${us.roundToInt()} us"
    us < 1e6 -> String.format(Locale.ROOT, "%.2f ms", us / 1e3)
    else -> String.format(Locale.ROOT, "%.2f s", us / 1e6)
}

// ai: Gain (where the camera lists one: a UVC camera's, the simulated camera's; in its units, dB where it names none),
// ai: live; its first step the camera's own (Auto) where it can set its gain by itself
@Composable
private fun GainField(r: ReceiveState, c: ReceiveState.Control) {
    val g = r.gain
    val unit = c.unit.ifEmpty { "dB" }
    val label = if (g == null && c.canAuto) "Auto" else String.format(Locale.ROOT, "%.1f %s", g ?: c.value, unit)
    val span = c.max - c.min
    val n = 100
    val first = if (c.canAuto) 1 else 0
    val x = if (g == null) (if (c.canAuto) 0 else first + ((c.value - c.min) / span * (n - first)).roundToInt()) else first + ((g - c.min) / span * (n - first)).roundToInt()
    Field("Gain", label) {
        Bar(x.toFloat(), 0f..n.toFloat(), n - 1) { v ->
            val k = v.roundToInt()
            if (k < first) r.gainTo(null) else r.gainTo(snap(c, c.min + span * (k - first) / (n - first)))
        }
    }
}

// ai: Developer Tools (the app's ReceiveAdvanced): Show statistics, Save replays and its runs; under Show statistics
// ai: the lab line and the readout (the app's: every monospace line there only with the statistics on)
@Composable
private fun ReceiveTools(r: ReceiveState) {
    Fields {
        Select("Show statistics", r.showStats, listOf(false to "Off", true to "On")) { r.chooseStats(it) }
        Select("Save replays", r.replays, listOf(false to "Off", true to "On")) { r.chooseReplays(it) }
        if (r.showStats) {
            for (run in r.runs) {
                CodeBlock(when (run.state) {
                    ReceiveState.RunState.Recording -> "${run.run}: recording"
                    ReceiveState.RunState.Ending -> "${run.run}: saving"
                    ReceiveState.RunState.Ready -> "${run.run}: ${run.frames} frames, ${String.format(Locale.ROOT, "%.0f", run.bytes / 1e6)} MB" +
                        if (run.why.isEmpty()) "" else " (${run.why})"
                    ReceiveState.RunState.Failed -> "${run.run}: ${run.why}"
                })
                if (run.state == ReceiveState.RunState.Ready) Btn("Show in folder", modifier = Modifier.fillMaxWidth()) { Received.show(run.dir) }
            }
            r.lab.let { if (it.isNotEmpty()) CodeBlock(it, 13) }
            CodeBlock(r.readout)
        }
    }
}

// ai: The camera's crop as the decoder reads it, grey, fitted into its box on the soft fill, square corners (the web's
// ai: #view); in 2:1 a red 1 dp line across the middle of its long side, where the receiver cuts the two halves (the
// ai: app's guide). Over it while the camera is off "Camera off" with its icon, and "Getting ready" with a spinner until
// ai: the receiver first answers (Readout.gettingReady).
@Composable
private fun Preview(r: ReceiveState, modifier: Modifier) {
    Box(modifier.clipToBounds().background(Soft), contentAlignment = Alignment.Center) {
        r.preview?.let { Image(it, contentDescription = null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit, filterQuality = FilterQuality.Low) }
        if (r.layout == "2:1") Canvas(Modifier.matchParentSize()) {
            val wide = size.width >= size.height
            val a = if (wide) Offset(size.width / 2, 0f) else Offset(0f, size.height / 2)
            val b = if (wide) Offset(size.width / 2, size.height) else Offset(size.width, size.height / 2)
            drawLine(Color(0xFFE53935), a, b, strokeWidth = 1.dp.toPx())
        }
        val off = r.phase == ReceiveState.Phase.Idle || r.phase is ReceiveState.Phase.Error
        val waiting = Readout.gettingReady(r.on, r.phase == ReceiveState.Phase.Starting, r.phase == ReceiveState.Phase.Loading, r.rx)
        if (off || waiting) Column(Modifier.matchParentSize().background(Soft), verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally) {
            if (off) Icon(Icons.cameraOff, contentDescription = null, Modifier.size(24.dp), tint = Muted) else Spinner()
            Text(if (off) "Camera off" else "Getting ready", style = MaterialTheme.typography.bodyMedium, color = Muted, textAlign = TextAlign.Center)
        }
    }
}
