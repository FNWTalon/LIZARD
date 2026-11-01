package dev.lizard.desktop

import androidx.compose.foundation.Image
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.lizard.readout.Readout
import dev.lizard.ui.Access
import dev.lizard.ui.Act
import dev.lizard.ui.Choice
import dev.lizard.ui.Code
import dev.lizard.ui.ColumnWidth
import dev.lizard.ui.FoldKey
import dev.lizard.ui.Folds
import dev.lizard.ui.Icons
import dev.lizard.ui.Press
import dev.lizard.ui.ReceiveHost
import dev.lizard.ui.RunState
import dev.lizard.ui.RxPhase
import dev.lizard.ui.Setting
import dev.lizard.ui.Slide
import dev.lizard.ui.Typed
import dev.lizard.ui.runLine
import java.util.Locale
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt
import dev.lizard.ui.Screen

// ai: The tool screens' folds and rail on the desktop, under the keys Prefs kept them by
class DesktopFolds(private val app: App) : Folds {
    private val r get() = app.recv
    private val s get() = app.send
    override fun open(k: FoldKey) = when (k) {
        FoldKey.RecvSettings -> r.settingsOpen; FoldKey.RecvExperimental -> r.experimentalOpen; FoldKey.RecvTools -> r.toolsOpen; FoldKey.RecvCollapsed -> r.collapsed
        FoldKey.SendSettings -> s.settingsOpen; FoldKey.SendExperimental -> s.experimentalOpen; FoldKey.SendTools -> s.toolsOpen; FoldKey.SendCollapsed -> s.collapsed
    }
    override fun toggle(k: FoldKey) = when (k) {
        FoldKey.RecvSettings -> r.toggleSettings(); FoldKey.RecvExperimental -> r.toggleExperimental(); FoldKey.RecvTools -> r.toggleTools(); FoldKey.RecvCollapsed -> r.toggleRail()
        FoldKey.SendSettings -> s.toggleSettings(); FoldKey.SendExperimental -> s.toggleExperimental(); FoldKey.SendTools -> s.toggleTools(); FoldKey.SendCollapsed -> s.toggleRail()
    }
}

// ai: The column's width on the desktop (2026-10-31: dragged as the phone's): the web's 20rem until its edge is dragged,
// ai: kept in Prefs (sideDp), one width for Receive and Send; its rows of buttons never wider than the default's (2026-11-01)
class DesktopColumn : ColumnWidth {
    override var dp by mutableStateOf(Prefs.sideDp)
    override var dragging by mutableStateOf(false)
    override val stretch = false
    override fun default(screen: Dp) = 320.dp
    override fun save() { Prefs.sideDp = dp }
}

// ai: Receive on the desktop (the screen is ../lizard-ui's ReceiveScreen): the receive session's state (ReceiveState),
// ai: the camera's settings as the camera lists them (its modes, its zoom, exposure and gain where it has them), a
// ai: phone by its address, an upload; the preview the session's crop pulled as a bitmap; a received file's Open and
// ai: Show in folder
class DesktopReceive(private val app: App) : ReceiveHost {
    private val r get() = app.recv
    override val folds = app.folds
    override val width = app.column
    override val canPortrait = false
    override val swapInColumn = true
    override fun goHome() = app.go(Screen.Home)
    override fun goSend() = app.go(Screen.Send)
    override val line get() = r.line
    override val showStats get() = r.showStats
    override val heat: String? = null
    override val hint get() = r.hint
    override val note get() = r.fileNote
    override val running get() = r.running
    override val starting get() = r.phase == RxPhase.Starting
    override fun toggleCamera() = r.toggle()
    override val access: Access? = null
    override val kept get() = r.kept?.let { k -> listOf(Act("Open", icon = Icons.open) { Received.open(k.file) }, Act("Show in folder") { Received.show(k.file) }) }
    override val keptName get() = r.kept?.name.orEmpty()
    override val keptLabels = listOf("Open", "Show in folder")
    // ai: the rate only while the figures are this stream's (ReceiveState.live), none with the camera off
    private val reading get() = r.on && ((r.rx.hasFile && !r.rx.verified) || r.testing)
    override val railRate get() = if (reading && r.live) r.rx.goodputKBs else null
    override val railPct get() = if (reading && !r.testing) Readout.fraction(r.rx) else null
    override fun cropRatio(portrait: Boolean) = r.cropRatio
    override val twoToOne get() = r.layout == "2:1"
    override val off get() = if (r.phase == RxPhase.Idle || r.phase is RxPhase.Error) "Camera off" else ""
    override val waiting get() = Readout.gettingReady(r.on, r.phase == RxPhase.Starting, r.phase == RxPhase.Loading, r.rx)
    // ai: the crop as the decoder reads it (Session.preview, grey), fitted into the box
    @Composable
    override fun Picture(modifier: Modifier, portrait: Boolean) {
        r.preview?.let { Image(it, contentDescription = null, modifier, contentScale = ContentScale.Fit, filterQuality = FilterQuality.Low) }
    }

    // ai: Settings (the web's #dev): a menu or a slider each, the camera's controls where it has them
    override val settings: List<Setting> get() {
        val ci = r.info
        val m = r.mode
        val out = mutableListOf<Setting>()
        // ai: GPU and CPU; with neither chosen the decoder is auto's and the one it runs on is shown
        out += Choice("Decoder", if (r.decoder == "auto") r.autoRan.ifEmpty { "GPU" }.lowercase() else r.decoder, listOf("gpu" to "GPU", "cpu" to "CPU")) { r.chooseDecoder(it) }
        out += Choice("Camera", r.camera.ifEmpty { r.cameras.firstOrNull()?.id ?: "" }, r.cameraOptions, blank = "Camera", onOpen = { r.refreshCameras() }) { r.chooseCameraOption(it) }
        // ai: a phone by its address (ReceiveState.askAddress): asked for, or the camera already one
        if (r.askAddress || r.typedPhone != null)
            out += Typed("Phone address", r.typedPhone?.removePrefix("phone:")?.removeSuffix(":${ReceiveState.PHONE_PORT}").orEmpty(), "IP address") { r.choosePhoneAddress(it) }
        // ai: the sizes and rates the camera offers, once one has been opened. None for a phone or an upload: their
        // ai: frames are a camera's crop already (cam/phone.cpp, upload.cpp), not a UVC device's modes
        if (ci != null && m != null && !r.fixed) {
            val sizes = ci.modes.distinctBy { it.size }.sortedBy { it.w.toLong() * it.h }
            out += Choice("Resolution", m.size, sizes.map { it.size to "${it.w} x ${it.h}" }) { r.chooseRes(it) }
            // ai: the pixel formats at this size, where there is a choice (a webcam's uncompressed one against its Motion-JPEG)
            val formats = ci.modes.filter { it.size == m.size }.map { it.format }.distinct()
            if (formats.size > 1) out += Choice("Format", m.format, formats.map { it to it }) { r.chooseFormat(it) }
            val rates = ci.modes.filter { it.size == m.size && it.format == m.format }.map { it.fps }.distinctBy { ReceiveState.fpsText(it) }.sorted()
            out += Choice("Frame rate", ReceiveState.fpsText(m.fps), rates.map { ReceiveState.fpsText(it) to ReceiveState.fpsText(it) }) { v ->
                rates.firstOrNull { ReceiveState.fpsText(it) == v }?.let { r.chooseFps(it) }
            }
        }
        // ai: a phone crops on its side, an upload was cropped when recorded (the receiver takes their crop): no Crop here
        if (!r.fixed) {
            out += Choice("Crop", r.layout, listOf("1:1" to "1:1", "2:1" to "2:1")) { r.chooseLayout(it) }
            ci?.control("zoom")?.let { out += zoom(it) }
            ci?.control("exposure")?.let { out += exposure(it) }
            ci?.control("gain")?.let { out += gain(it) }
        }
        // ai: the most frames a GPU launch takes: none for the C, which decodes a frame at a time
        if (!r.cpu) out += Slide("Batch size", "${r.batch}", r.batch.toFloat(), 1f..32f, 30, done = { r.commitBatch() }) { r.chooseBatch(it.roundToInt()) }
        return out
    }

    // ai: Zoom (UVC zoom_absolute), live: shown as a ratio of its least ("1.5x") where it starts above 0
    private fun zoom(c: ReceiveState.Control): Slide {
        val z = (r.zoom ?: c.value).coerceIn(c.min, c.max)
        val label = when {
            c.unit == "x" -> String.format(Locale.ROOT, "%.1fx", z)
            c.min > 0 -> String.format(Locale.ROOT, "%.1fx", z / c.min)
            else -> String.format(Locale.ROOT, "%.0f", z)
        }
        return Slide("Zoom", label, z.toFloat(), c.min.toFloat()..c.max.toFloat(), stepsOf(c)) { r.zoomTo(snap(c, it.toDouble())) }
    }

    // ai: Exposure in us on a log scale (a camera's range spans decades: 10 us to seconds), live; its first step the
    // ai: camera's own (Auto) where the camera can expose by itself, which is the default
    private fun exposure(c: ReceiveState.Control): Slide {
        val lo = maxOf(c.min, 1.0); val hi = maxOf(c.max, lo + 1)
        val first = if (c.canAuto) 1 else 0
        val n = EXPOSURE_STEPS
        fun usAt(x: Int) = snap(c, lo * exp(ln(hi / lo) * (x - first).toDouble() / (n - first)))
        fun xOf(us: Double) = first + ((ln(us.coerceIn(lo, hi) / lo) / ln(hi / lo)) * (n - first)).roundToInt()
        val us = r.exposure
        val x = if (us == null) (if (c.canAuto) 0 else xOf(c.value)) else xOf(us)
        val label = if (us == null && c.canAuto) "Auto" else usText(us ?: c.value)
        return Slide("Exposure", label, x.toFloat(), 0f..n.toFloat(), n - 1) { v ->
            val k = v.roundToInt()
            if (k < first) r.exposureTo(null) else r.exposureTo(usAt(k))
        }
    }

    // ai: Gain (where the camera lists one: a UVC camera's, the simulated camera's; in its units, dB where it names none),
    // ai: live; its first step the camera's own (Auto) where it can set its gain by itself
    private fun gain(c: ReceiveState.Control): Slide {
        val g = r.gain
        val unit = c.unit.ifEmpty { "dB" }
        val label = if (g == null && c.canAuto) "Auto" else String.format(Locale.ROOT, "%.1f %s", g ?: c.value, unit)
        val span = c.max - c.min
        val n = 100
        val first = if (c.canAuto) 1 else 0
        val x = if (g == null) (if (c.canAuto) 0 else first + ((c.value - c.min) / span * (n - first)).roundToInt()) else first + ((g - c.min) / span * (n - first)).roundToInt()
        return Slide("Gain", label, x.toFloat(), 0f..n.toFloat(), n - 1) { v ->
            val k = v.roundToInt()
            if (k < first) r.gainTo(null) else r.gainTo(snap(c, c.min + span * (k - first) / (n - first)))
        }
    }

    override val experimental: List<Setting> get() = emptyList()

    // ai: Developer Tools: Show statistics, Save replays and its runs; under Show statistics the runs, the lab line and the
    // ai: readout
    override val tools: List<Setting> get() {
        val out = mutableListOf<Setting>()
        out += Choice("Show statistics", r.showStats, listOf(false to "Off", true to "On")) { r.chooseStats(it) }
        // ai: none for an upload: it is a replay already
        if (!r.upload) out += Choice("Save replays", r.replays, listOf(false to "Off", true to "On")) { r.chooseReplays(it) }
        if (r.showStats) {
            for (run in r.runs) {
                out += Code(runLine(run), 12)
                if (run.state == RunState.Ready) out += Press(Act("Show in folder") { Received.show(run.file) })
            }
            r.lab.let { if (it.isNotEmpty()) out += Code(it, 13) }
            out += Code(r.readout, 12)
        }
        return out
    }
    override val about get() = app.about

    companion object {
        private const val EXPOSURE_STEPS = 100
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
        private fun usText(us: Double) = when {
            us < 1000 -> "${us.roundToInt()} us"
            us < 1e6 -> String.format(Locale.ROOT, "%.2f ms", us / 1e3)
            else -> String.format(Locale.ROOT, "%.2f s", us / 1e6)
        }
    }
}
