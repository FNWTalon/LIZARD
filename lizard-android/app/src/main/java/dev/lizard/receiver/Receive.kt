package dev.lizard.receiver

import android.graphics.Matrix
import android.graphics.RectF
import android.view.Surface
import android.view.TextureView
import android.view.WindowManager
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.requiredSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import dev.lizard.readout.Readout
import dev.lizard.ui.Access
import dev.lizard.ui.Act
import dev.lizard.ui.Choice
import dev.lizard.ui.Code
import dev.lizard.ui.ColumnWidth
import dev.lizard.ui.FoldKey
import dev.lizard.ui.Folds
import dev.lizard.ui.Icons
import dev.lizard.ui.Lines
import dev.lizard.ui.Press
import dev.lizard.ui.ReceiveHost
import dev.lizard.ui.RunState
import dev.lizard.ui.RxPhase
import dev.lizard.ui.Setting
import dev.lizard.ui.Slide
import dev.lizard.ui.flag
import dev.lizard.ui.num
import dev.lizard.ui.obj
import dev.lizard.ui.runLine
import dev.lizard.ui.str
import java.util.Locale
import kotlin.math.roundToInt
import dev.lizard.ui.Screen

// ai: The tool screens' folds and rail on the phone, under the keys the preferences kept them by (MainActivity.FOLDS)
internal class AndroidFolds(private val a: MainActivity) : Folds {
    private fun key(k: FoldKey) = when (k) {
        FoldKey.RecvSettings -> "recvSettings"; FoldKey.RecvExperimental -> "recvExperimental"; FoldKey.RecvTools -> "recvAdvanced"
        FoldKey.RecvCollapsed -> "recvCollapsed"; FoldKey.SendSettings -> "sendSettings"; FoldKey.SendExperimental -> "sendExperimental"
        FoldKey.SendTools -> "sendAdvanced"; FoldKey.SendCollapsed -> "sendCollapsed"
    }
    override fun open(k: FoldKey) = a.isOpen(key(k))
    override fun toggle(k: FoldKey) = a.toggle(key(k))
}

// ai: The landscape column's width on the phone (2026-10-05): 240 dp or a third of the screen's width where that is less
// ai: (320 and 0.45 until then, a fourth wider), or what its edge was dragged to (MainActivity.sideDp), one width for
// ai: Receive and Send
internal class AndroidColumn(private val a: MainActivity) : ColumnWidth {
    override var dp: Float get() = a.sideDp; set(v) { a.sideDp = v }
    override var dragging: Boolean get() = a.sideDragging; set(v) { a.sideDragging = v }
    override fun default(screen: Dp) = minOf(240.dp, screen * 0.34f)
    override fun save() = a.saveSide()
}

// ai: Receive on the phone (the screen is ../lizard-ui's ReceiveScreen): the receive session's state (ReceiveState, the
// ai: activity's poll), the camera's permission, the heat, the PC link, the lens's settings; the preview the camera's
// ai: second stream in a TextureView; a received file's Open, Share and Save
internal class AndroidReceive(private val a: MainActivity) : ReceiveHost {
    private val r get() = a.receive
    private val s get() = a.settings
    override val folds = AndroidFolds(a)
    override val width = AndroidColumn(a)
    override val canPortrait = true
    override val swapInColumn = false
    override fun goHome() = a.go(Screen.Home)
    override fun goSend() = a.go(Screen.Send)
    private val on get() = a.phase == RxPhase.On
    override val line: Readout.Line get() =
        if (!a.granted) Readout.Line(if (a.denied) "Camera access is off." else "")
        else Readout.line(on, a.phase == RxPhase.Starting, a.phase == RxPhase.Loading, (a.phase as? RxPhase.Error)?.why, a.rx,
            a.files.firstOrNull { it.root == a.root && it.bundle }?.files ?: 0)
    override val showStats get() = s.stats == "on"
    override val heat get() = heatText(a.heat, a.clocks, r.cameraSlow)
    override val hint get() = a.hint
    override val note get() = a.note
    override val running get() = on || a.phase == RxPhase.Starting || a.phase == RxPhase.Loading
    override val starting get() = a.phase == RxPhase.Starting
    override fun toggleCamera() = a.toggleCamera()
    override val access get() = if (a.granted) null else Access(Act(if (a.denied) "Open settings" else "Allow camera") { if (a.denied) a.openAppSettings() else a.askCamera() })
    // ai: the received file, the test stream's figures in its place while the camera reads that (2026-10-05)
    private val testing get() = on && a.rx.state == "test"
    private val keptEntry get() = if (a.rx.verified && !testing) a.files.firstOrNull { it.root == a.root } else null
    override val kept get() = keptEntry?.let { e ->
        listOf(Act("Open", icon = Icons.open) { a.open(e) }, Act("Share") { a.share(e) }, Act("Save", icon = Icons.save) { a.saveCopy(e) })
    }
    override val keptName get() = keptEntry?.name.orEmpty()
    override val keptLabels = listOf("Open", "Share", "Save")
    private val reading get() = on && ((a.rx.hasFile && !a.rx.verified) || testing)
    override val railRate get() = if (reading) a.rx.goodputKBs else null
    override val railPct get() = if (reading && !testing) Readout.fraction(a.rx) else null
    override fun cropRatio(portrait: Boolean) = cropOf(portrait).ratio
    override val twoToOne get() = s.layout == "2:1"
    override val off get() = when {
        !a.granted -> "No camera yet"
        a.phase == RxPhase.Idle || a.phase is RxPhase.Error -> "Camera off"
        else -> ""
    }
    override val waiting get() = a.granted && Readout.gettingReady(on, a.phase == RxPhase.Starting, a.phase == RxPhase.Loading, a.rx)

    // ai: the crop the decoder reads, as the screen shows it: 1:1 the frame's centre square, 2:1 its centre region twice
    // ai: as long as high along the long side (rx/receiver.cpp push). The sensor is landscape and the camera turns the
    // ai: preview upright, so a portrait screen shows the long side vertically. fullX / cropX is how much wider than
    // ai: the crop the whole preview is, and the same for the height; ratio is the crop's width over its height.
    private class Crop(val fullX: Float, val fullY: Float, val cropX: Float, val cropY: Float) { val ratio get() = cropX / cropY }
    private fun cropOf(portrait: Boolean): Crop {
        val c = a.cam
        val w = (c?.size?.width ?: 16).toFloat()
        val h = (c?.size?.height ?: 9).toFloat()
        val lo = minOf(w, h)
        val hi = maxOf(w, h)
        val two = s.layout == "2:1"
        val side = if (two) minOf(lo, hi / 2) else lo
        val cropLong = if (two) 2 * side else side
        return Crop(if (portrait) lo else hi, if (portrait) hi else lo, if (portrait) side else cropLong, if (portrait) cropLong else side)
    }

    // ai: Only the crop is shown, as the web receiver shows its crop: the whole preview (the camera's second stream, written
    // ai: by the receive session's camera, liblizard/core/cam/android.cpp, into the TextureView's surface) is laid out at
    // ai: the size that puts the crop over the box, and clipped.
    @Composable
    override fun Picture(modifier: Modifier, portrait: Boolean) = BoxWithConstraints(modifier.clipToBounds(), contentAlignment = Alignment.Center) {
        val g = cropOf(portrait)
        val c = a.cam
        val pw = (c?.preview?.width ?: 16).toFloat()
        val ph = (c?.preview?.height ?: 9).toFloat()
        // ai: the layout listener turns by the preview's current shape, not the one the view was made with (2026-10-07):
        // ai: the factory runs once, and a preview of another shape would be turned by the old one at every layout
        val shape = rememberUpdatedState(pw to ph)
        val full = Modifier.requiredSize(maxWidth * (g.fullX / g.cropX), maxHeight * (g.fullY / g.cropY))
        AndroidView({ ctx ->
            TextureView(ctx).apply {
                surfaceTextureListener = r.textureListener
                addOnLayoutChangeListener { v, _, _, _, _, _, _, _, _ -> val (w, hh) = shape.value; turn(v as TextureView, w, hh) }
            }
        }, full, update = { turn(it, pw, ph) })
    }

    // ai: The preview upright on a turned display (Google's Camera2 sample's configureTransform): the camera turns its
    // ai: buffers for the display's natural orientation only; at 90 or 270 degrees the view maps onto the buffer's turned
    // ai: rectangle, scaled back to the view and rotated against the display.
    private fun turn(v: TextureView, pw: Float, ph: Float) {
        val vw = v.width.toFloat()
        val vh = v.height.toFloat()
        if (vw == 0f || vh == 0f) return
        @Suppress("DEPRECATION")
        val rot = (a.getSystemService(android.content.Context.WINDOW_SERVICE) as WindowManager).defaultDisplay.rotation
        val m = Matrix()
        val view = RectF(0f, 0f, vw, vh)
        val cx = view.centerX()
        val cy = view.centerY()
        if (rot == Surface.ROTATION_90 || rot == Surface.ROTATION_270) {
            val buf = RectF(0f, 0f, ph, pw)
            buf.offset(cx - buf.centerX(), cy - buf.centerY())
            m.setRectToRect(view, buf, Matrix.ScaleToFit.FILL)
            val k = maxOf(vh / ph, vw / pw)
            m.postScale(k, k, cx, cy)
            m.postRotate(90f * (rot - 2), cx, cy)
        } else if (rot == Surface.ROTATION_180) m.postRotate(180f, cx, cy)
        v.setTransform(m)
    }

    // ai: Settings (2026-10-02, the web's): the decoder (auto shows what it runs on; PC only while a PC receiver is heard
    // ai: on the link), the PC link while a cable is in, the PC decoder's Compression, the GPU's Batch size, the camera
    // ai: (the back lenses), its mode (Resolution, Format and Frame rate, the session's), Zoom, Crop
    override val settings: List<Setting> get() {
        val out = mutableListOf<Setting>()
        val sizes = r.modes.map { it.size }.toSet()
        val res = r.mode?.size ?: "1920x1080"
        val fmt = r.mode?.format ?: "auto"
        val decoders = listOf("gpu" to "GPU", "cpu" to "CPU") + listOfNotNull(("pc" to "PC").takeIf { a.pcPresent || s.decoder == "pc" })
        out += Choice("Decoder", if (s.decoder == "auto") a.autoRan.ifEmpty { "GPU" }.lowercase() else s.decoder, decoders) { a.change(s.copy(decoder = it)) }
        if (a.usb.connected) out += pcLink()
        // ai: the PC decoder's frames packed as .lizard frames or raw (Settings.pcPack): raw for a phone too weak or too warm
        if (s.decoder == "pc") out += Choice("Compression", s.pcPack, listOf("on" to "On", "off" to "Off")) { a.change(s.copy(pcPack = it)) }
        // ai: the most frames a GPU launch takes (Settings.batch); the C decodes a frame at a time: not shown with it
        if (s.decoder != "cpu" && s.decoder != "pc") out += Slide("Batch size", "${s.frames}", s.frames.toFloat(), 1f..32f, 30) { v ->
            val n = v.roundToInt().coerceIn(1, 32); if (n != s.frames) a.change(s.copy(batch = n.toString()))
        }
        // ai: the back cameras by lens (the closest-focusing where none is chosen: the session's first listed)
        out += Choice("Camera", s.camera.ifEmpty { r.camId }, r.lenses.map { it.id to it.label }, blank = "Camera") { a.change(s.copy(camera = it)) }
        out += Choice("Resolution", res, Settings.RESOLUTIONS.map { it to it }, enabledFor = { sizes.isEmpty() || it in sizes }) { r.chooseResolution(it) }
        // ai: the camera's format: Auto, or each the lens offers at this resolution (RAW at any) that the decoder reads
        // ai: (Private with the GPU decoder alone); shown where there is a choice
        val fmts = Settings.FORMATS.filter { (k, _) ->
            r.modes.any { it.format == k && (k.startsWith("raw") || it.size == res) } && !(k == "private" && (s.decoder == "cpu" || s.decoder == "pc"))
        }
        if (fmts.size > 1 || fmt != "auto") out += Choice("Format", fmt, listOf("auto" to "Auto") + fmts) { r.chooseFormat(it) }
        // ai: the rates the lens offers at this size and format, up to 60 (cam/android.cpp ratesOf), where there is a choice
        val rates = r.modes.filter { it.size == res && it.format == fmt }.map { it.fps }.distinct().sortedDescending()
        val fps = r.mode?.fps ?: rates.firstOrNull() ?: 0.0
        if (rates.size > 1) out += Choice("Frame rate", fpsText(fps), rates.map { fpsText(it) to fpsText(it) }) { v ->
            rates.firstOrNull { fpsText(it) == v }?.let { r.chooseFps(it) }
        }
        out += zoom(r.zoomRange)
        out += Choice("Crop", s.layout, Settings.LAYOUTS.map { it to it }) { a.change(s.copy(layout = it)) }
        return out
    }

    // ai: Zoom as a slider (2026-10-01): 0.1x steps over the camera's whole range (the S26's camera 0 runs 0.6 to 10x and
    // ai: reaches its 3x and 5x lenses by the zoom), moved live on the running camera while it slides and kept at every step
    private fun zoom(range: ClosedFloatingPointRange<Float>?): Slide {
        val lo = range?.start ?: 1f
        val hi = maxOf(lo + 0.1f, range?.endInclusive ?: 4f)
        val z = (r.zoom ?: 1.5f).coerceIn(lo, hi)
        return Slide("Zoom", "%.1fx".format(Locale.ROOT, z), z, lo..hi, maxOf(0, ((hi - lo) * 10).roundToInt() - 1)) { v ->
            val n = (v * 10).roundToInt() / 10f; if (n != r.zoom) r.zoomTo(n)
        }
    }

    // ai: The PC link, while a cable is in (the core's linkState): a PC receiver heard, its address ("USB" for the
    // ai: accessory) and, on the PC decoder with the camera on, the frames a second it carries; else "disconnected", the
    // ai: USB accessory where it is open and the tether's address, and the tethering page where the cable has neither
    private fun pcLink(): Lines {
        val lk = a.link
        val st = a.stats?.obj("stream")
        val addrs = PcLink.usbAddresses()
        val accessory = lk?.flag("accessory") == true
        val port = lk?.takeIf { "port" in it }?.num("port")?.toInt() ?: 47230
        val present = a.pcPresent
        val lines = if (present) listOfNotNull(st?.takeIf { s.decoder == "pc" && a.phase == RxPhase.On }?.let {
                String.format(Locale.ROOT, "%.0f of %.0f frames a second", it.num("sentFps"), it.num("cameraFps")) })
            else (listOfNotNull("USB accessory".takeIf { accessory }) + addrs.map { "USB $it:$port" }).ifEmpty { listOf("USB tethering off") }
        val tether = if (!a.usb.tether && !accessory && !present && addrs.isEmpty()) Act("USB tethering") { PcLink.openTethering(a) } else null
        return Lines("PC link", if (present) "to ${lk?.str("peer").orEmpty()}" else "disconnected", lines, tether)
    }

    override val experimental: List<Setting> get() = emptyList()

    // ai: Developer Tools: Show statistics, then Save replays (none for the PC decoder: the PC's own records them) and its
    // ai: runs, each with its Download once ended; under Show statistics every monospace line (a run's, the note after a
    // ai: Download, the lab line, the readout)
    override val tools: List<Setting> get() {
        val out = mutableListOf<Setting>()
        out += Choice("Show statistics", s.stats, listOf("off" to "Off", "on" to "On")) { a.change(s.copy(stats = it)) }
        if (s.decoder != "pc") out += Choice("Save replays", s.replays, listOf("off" to "Off", "on" to "On")) { a.change(s.copy(replays = it)) }
        for (rp in r.runs) {
            if (s.stats == "on") out += Code(runLine(rp), 12)
            if (rp.state == RunState.Ready) out += Press(Act(if (MainActivity.exporting == rp.run) "Downloading" else "Download") { a.downloadReplay(rp) },
                enabled = MainActivity.exporting.isEmpty())
        }
        if (s.stats == "on" && MainActivity.replayNote.isNotEmpty()) out += Code(MainActivity.replayNote, 12)
        if (s.stats == "on") {
            val lab = Readout.lab(a.phase == RxPhase.On, a.rx)
            if (lab.isNotEmpty()) out += Code(lab)
            out += Code(a.devReadout(), 12)
        }
        return out
    }
    override val about get() = a.aboutLine()
}

// ai: a rate as the Frame rate menu shows it: whole where it is, else two places (the desktop's ReceiveState.fpsText)
private fun fpsText(v: Double) =
    if (kotlin.math.abs(v - v.roundToInt()) < 0.01) "${v.roundToInt()}" else String.format(java.util.Locale.ROOT, "%.2f", v)
