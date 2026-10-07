package dev.lizard.desktop

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonObject
import java.awt.BasicStroke
import java.awt.Canvas
import java.awt.FileDialog
import java.awt.Font
import java.awt.Frame
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.datatransfer.DataFlavor
import java.awt.dnd.DnDConstants
import java.awt.dnd.DropTarget
import java.awt.dnd.DropTargetAdapter
import java.awt.dnd.DropTargetDropEvent
import java.awt.geom.Path2D
import javax.swing.SwingUtilities
import java.awt.Cursor
import java.awt.EventQueue
import java.awt.Graphics
import java.awt.Point
import java.awt.Toolkit
import java.awt.event.ComponentAdapter
import java.awt.event.ComponentEvent
import java.awt.event.HierarchyEvent
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import javax.swing.Timer
import kotlin.concurrent.thread
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt

// ai: The code area (the web's <main>): a heavyweight AWT canvas, a native window of its own inside the Compose window
// ai: (Screen.kt's SwingPanel), on which the presenter's swapchain draws; white, and nothing painted over the
// ai: presenter's pictures by AWT. Its size in device px is the room the format is picked from; it tells the state when
// ai: it has a window (addNotify), is shown or resized, and, before that window goes (removeNotify: a dispose, a window
// ai: made again), stops the presenter on it. A click on it leaves full screen, as a tap on the web's code does; the
// ai: cursor is hidden there. Idle (2026-10-07, the web's hint in the middle of its page and the app's code box) it
// ai: is the file's area: the soft fill, the file icon and "Choose a file", or the file's name once one is chosen, or
// ai: "Test stream"; a click opens the file dialog and a file dropped on it is taken (the test stream off). AWT paints
// ai: that itself, since nothing of Compose can be drawn over a heavyweight canvas.
class CodeCanvas(private val s: SendState) : Canvas() {
    @Volatile var presenting = false

    init {
        background = java.awt.Color.WHITE
        isFocusable = false
        addComponentListener(object : ComponentAdapter() { override fun componentResized(e: ComponentEvent) = s.areaChanged() })
        // ai: shown or hidden with its parents (Compose's SwingPanel sizes it before it shows it)
        addHierarchyListener { e -> if (e.changeFlags and HierarchyEvent.SHOWING_CHANGED.toLong() != 0L) s.areaChanged() }
        addMouseListener(object : MouseAdapter() { override fun mouseClicked(e: MouseEvent) { if (s.full) s.toggleFull() else if (chooses()) s.chooseFile() } })
        dropTarget = DropTarget(this, object : DropTargetAdapter() {
            override fun drop(e: DropTargetDropEvent) {
                if (s.sending) { e.rejectDrop(); return }
                e.acceptDrop(DnDConstants.ACTION_COPY)
                val taken = runCatching {
                    @Suppress("UNCHECKED_CAST")
                    val files = e.transferable.getTransferData(DataFlavor.javaFileListFlavor) as List<File>
                    files.firstOrNull { it.isFile }?.let { if (s.test) s.chooseTest(false); s.pick(it); true } ?: false
                }.getOrDefault(false)
                e.dropComplete(taken)
            }
        })
    }

    private fun chooses() = !s.sending && !s.test

    // ai: the look for the state: a hand over the file's area, the default cursor otherwise (full screen hides it)
    fun idleLook() {
        if (!s.full) cursor = if (chooses()) Cursor.getPredefinedCursor(Cursor.HAND_CURSOR) else Cursor.getDefaultCursor()
        repaint()
    }

    override fun update(g: Graphics) = paint(g)
    override fun paint(g: Graphics) {
        if (presenting) return
        val g2 = g as Graphics2D
        g2.color = java.awt.Color(0xF1, 0xF1, 0xF1)   // ai: the soft fill (Parts.kt Soft)
        g2.fillRect(0, 0, width, height)
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
        g2.color = java.awt.Color(0x66, 0x66, 0x66)   // ai: Muted
        // ai: the file icon (Parts.kt Icons.file, Feather's on a 24 grid, 2 px strokes), then the line 8 px under it
        val cx = width / 2.0; val cy = height / 2.0 - 16
        val icon = Path2D.Double().apply {
            moveTo(14.0, 2.0); lineTo(6.0, 2.0); curveTo(4.9, 2.0, 4.0, 2.9, 4.0, 4.0); lineTo(4.0, 20.0); curveTo(4.0, 21.1, 4.9, 22.0, 6.0, 22.0)
            lineTo(18.0, 22.0); curveTo(19.1, 22.0, 20.0, 21.1, 20.0, 20.0); lineTo(20.0, 8.0); closePath()
            moveTo(14.0, 2.0); lineTo(14.0, 8.0); lineTo(20.0, 8.0)
        }
        g2.stroke = BasicStroke(2f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
        val t = g2.transform
        g2.translate(cx - 12, cy - 12); g2.draw(icon); g2.transform = t
        val text = if (s.test) "Test stream" else s.file?.name ?: "Choose a file"
        g2.font = Font(Font.SANS_SERIF, Font.PLAIN, 14)
        val fm = g2.fontMetrics
        g2.drawString(text, (cx - fm.stringWidth(text) / 2.0).toFloat(), (cy + 12 + 8 + fm.ascent).toFloat())
    }
    override fun addNotify() { super.addNotify(); EventQueue.invokeLater { s.areaChanged() } }
    override fun removeNotify() { s.canvasGone(); super.removeNotify() }

    // ai: the canvas's size in device px (its user-space size by the screen's scale)
    fun devicePx(): Pair<Int, Int> {
        val t = graphicsConfiguration?.defaultTransform
        return (width * (t?.scaleX ?: 1.0)).roundToInt() to (height * (t?.scaleY ?: 1.0)).roundToInt()
    }

    fun hideCursor(on: Boolean) { cursor = if (on) blank else Cursor.getDefaultCursor() }
    private val blank by lazy { Toolkit.getDefaultToolkit().createCustomCursor(BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB), Point(0, 0), "none") }
}

// ai: The run (the web's send.mjs, lizard-android/.../Send.kt SendState): the settings, kept (Prefs), the sender and the
// ai: presenter, and what the column shows. Start makes the sender (txCreate; the GPU painter's txPrepare off the UI
// ai: thread where the encoder is not the CPU), then, once the code area has a size and is shown, picks the format from
// ai: its room, configures, and starts the presenter on the canvas. A change of the area's size,
// ai: Size, Gap, Ring, Codes or Blocks re-picks, and configures where the format changed (a configure drops the frames
// ai: painted ahead; the transfer goes on, as the web's re-pick); Pictures a second configures (the word states it) and
// ai: tells the presenter. Every Native call here is on the UI thread but txCreate and txPrepare; the native side holds
// ai: one lock for configure and the presenter's take.
class SendState {
    sealed interface Phase { data object Idle : Phase; data object Preparing : Phase; data object On : Phase; data class Error(val why: String) : Phase }
    // ai: what is painted: the format configured
    data class Format(val n: Int, val subch: Int, val span: Int, val fps: Int, val codes: Int, val gap: Int, val painter: Int)

    var file by mutableStateOf<File?>(null)
    var fileSize by mutableStateOf(0L)
    // ai: the file's bytes as they go, every chunk's zstd frame or its own (the sender's stats, 2026-10-05); 0 before a send
    val sentBytes: Long get() = sender?.num("sentBytes")?.toLong() ?: 0L
    var test by mutableStateOf(Prefs.payload == "test")
    var enc by mutableStateOf(Prefs.enc.let { if (it == "gpu" || it == "cpu") it else "auto" })
    // ai: the painter the sender runs on now, "gpu" or "cpu" ("" before a send): the Encoder chip auto marks (2026-10-05)
    val painterNow: String get() = sender?.str("painter") ?: ""
    var blocks by mutableStateOf(Prefs.blocks.coerceIn(0, 128))
    var fps by mutableStateOf(Prefs.fps.coerceIn(1, 60))
    var size by mutableStateOf(Prefs.size.coerceIn(25, 100))
    var gap by mutableStateOf(Prefs.gap.coerceIn(0, 64))
    var ring by mutableStateOf(Prefs.ring.let { if (it in 0..3) it else -1 })
    var codes by mutableStateOf(if (Prefs.codes == 2) 2 else 1)
    var settingsOpen by mutableStateOf(Prefs.settingsOpen)
    var toolsOpen by mutableStateOf(Prefs.toolsOpen)
    var collapsed by mutableStateOf(Prefs.collapsed)

    var phase by mutableStateOf<Phase>(Phase.Idle)
    var paused by mutableStateOf(false)
    // ai: full screen (the button, F11; Escape or a click on the code leaves): the window full screen, the column folded
    // ai: to the rail unless opened there (fullOpen), the compositor bypassed
    var full by mutableStateOf(false)
    var fullOpen by mutableStateOf(false)
    var areaW by mutableStateOf(0)
    var areaH by mutableStateOf(0)
    var fmt by mutableStateOf<Format?>(null)
    var present by mutableStateOf<JsonObject?>(null)
    var sender by mutableStateOf<JsonObject?>(null)
    // ai: the stats as the native side wrote them, the last second's (the test hook prints them)
    var presentRaw = ""
        private set
    var senderRaw = ""
        private set
    var loadError = ""

    val canvas = CodeCanvas(this)
    private var h = 0L
    private var p = 0L
    private var gen = 0
    private var gpuWhy = ""
    // ai: the GPU painter made for this run (txPrepare), and a prepare under way on a thread of its own: while one is,
    // ai: stop leaves the sender for that thread to destroy (it holds the sender's lock inside txPrepare)
    private var prepared = false
    private var preparing = 0L
    private var runTest = false
    private var runName = ""
    private val timer = Timer(1000) { poll() }

    val presenting get() = p != 0L
    val sending get() = phase == Phase.On || phase == Phase.Preparing
    val canStart get() = loadError.isEmpty() && !sending && (test || file != null)
    val railShown get() = if (full) !fullOpen else collapsed
    val ringIndex get() = if (ring in 0..3) ring else Pick.RING_DEFAULT
    // ai: a symbol's room in the area, device px, the Size slider's share of it (send.mjs room)
    val room get() = Pick.room(areaW, areaH, codes, ringIndex, gap) * size / 100.0
    // ai: what auto takes in the area now (0 while it has no size): the configured format's while sending
    val autoBlocks get() = fmt?.takeIf { blocks == 0 }?.let { it.subch / 8 } ?: if (areaW > 0 && areaH > 0) Pick.pick(room, ringIndex) / 8 else 0
    // ai: what the code offers at most, KB/s: codes x blocks a frame x 469 B x the pictures a second it can show, the
    // ai: asked rate or the measured refresh where lower, so a limit outside the sender shows in the figure
    // ai: (2026-10-04); null before a configure
    val capacityKBs: Double? get() {
        val f = fmt ?: return null
        val hz = present?.num("hz") ?: 0.0
        return f.codes * (f.subch / 8) * 469.0 * (if (hz > 0) min(f.fps.toDouble(), hz) else f.fps.toDouble()) / 1000
    }

    // ai: the platform's own file dialog (GTK's on Linux), one file, over the sender's window (the canvas's frame)
    fun chooseFile() {
        val d = FileDialog(SwingUtilities.getWindowAncestor(canvas) as? Frame, "Choose a file", FileDialog.LOAD)
        d.isMultipleMode = false
        d.isVisible = true
        d.files.firstOrNull()?.let { pick(it) }
    }

    fun pick(f: File) {
        file = f; fileSize = f.length()
        if (phase is Phase.Error) phase = Phase.Idle
    }

    // ai: Start as the person presses it (the column's Start, the rail's play): before the first send, the brightness tip
    // ai: (2026-10-04; the web sender's the same): the monitor's contrast and brightness are the channel's and no app
    // ai: sets them (the contrast at its maximum took 2:1 past 2.90 MB/s). Shown until a send goes from it (its Start),
    // ai: Cancel leaving it for the next press; the test hook's start() and Resume never show it.
    var askBrightness by mutableStateOf(false)
    // ai: taken in this run, whether or not Prefs keep it (send.mjs seenHere)
    private var tipTaken = false
    fun requestStart() {
        if (!canStart) return
        if (tipTaken || Prefs.brightnessSeen) start() else askBrightness = true
    }
    fun answerBrightness(go: Boolean) {
        askBrightness = false
        if (go) { tipTaken = true; Prefs.brightnessSeen = true; start() }
    }

    fun start() {
        if (!canStart) return
        val f = file
        runTest = test
        runName = if (test) "the test stream" else f!!.name
        val path = if (test) "" else f!!.path
        val type = if (test) "" else runCatching { Files.probeContentType(f!!.toPath()) }.getOrNull() ?: ""
        val prepare = enc != "cpu"
        val my = ++gen
        phase = Phase.Preparing; paused = false
        thread(name = "lizard-send-prep", isDaemon = true) {
            val hnd = Native.txCreate(path, if (runTest) "" else runName, type)
            val err = if (hnd == 0L) Native.txError() else ""
            // ai: the GPU painter's device and pipelines made here, off the UI thread that configures
            val why = if (hnd != 0L && prepare) Native.txPrepare(hnd, Native.assets) else ""
            EventQueue.invokeLater {
                if (my != gen) { if (hnd != 0L) Native.txDestroy(hnd); return@invokeLater }   // ai: stopped meanwhile
                if (hnd == 0L) { phase = Phase.Error("This file cannot be sent: $err"); return@invokeLater }
                h = hnd; gpuWhy = why; prepared = prepare; fmt = null; phase = Phase.On
                if (why.isNotEmpty()) System.err.println("send: no GPU painter: $why")
                areaChanged()
            }
        }
    }

    fun stop() {
        gen++
        timer.stop()
        settle.stop()
        if (p != 0L) { Native.presentStop(p); p = 0L }
        canvas.presenting = false
        canvas.repaint()
        if (h != 0L && h != preparing) Native.txDestroy(h)
        h = 0L
        if (sending) phase = Phase.Idle
        paused = false; fmt = null; present = null; sender = null
    }

    private fun fail(why: String) { stop(); phase = Phase.Error(why) }

    // ai: Paused: the last picture stays (the presenter takes no new one); a receiver holds what it has.
    fun pause(on: Boolean) {
        if (phase != Phase.On) return
        paused = on
        if (p != 0L) Native.presentPause(p, on)
    }

    fun toggleFull() {
        full = !full; fullOpen = false
        canvas.hideCursor(full)
        if (p != 0L) Native.presentFullscreen(p, full)
    }

    fun toggleRail() { if (full) fullOpen = !fullOpen else { collapsed = !collapsed; Prefs.collapsed = collapsed } }
    fun toggleSettings() { settingsOpen = !settingsOpen; Prefs.settingsOpen = settingsOpen }
    fun toggleTools() { toolsOpen = !toolsOpen; Prefs.toolsOpen = toolsOpen }

    // ai: the settings: each kept as it moves; a slider's committed on release (commit), a chip's at once
    fun chooseTest(on: Boolean) { test = on; Prefs.payload = if (on) "test" else "file" }
    // ai: The encoder moved while sending: to the CPU, or to the GPU or auto once its painter is made, configured at once;
    // ai: else the GPU painter made first on a thread of its own (txPrepare), then configured. Made inside configure, on
    // ai: the UI thread, it froze the window for the driver's compile and held the sender's lock (2026-10-04).
    fun chooseEnc(v: String) {
        enc = v; Prefs.enc = v
        if (phase != Phase.On) return
        if (v == "cpu" || prepared) { configure(); return }
        if (preparing != 0L) return   // ai: the one under way configures with the encoder as it is then
        val hnd = h
        val my = gen
        preparing = hnd
        thread(name = "lizard-send-prep", isDaemon = true) {
            val why = Native.txPrepare(hnd, Native.assets)
            EventQueue.invokeLater {
                preparing = 0L
                if (my != gen || h != hnd) { Native.txDestroy(hnd); return@invokeLater }   // ai: stopped meanwhile: it is this thread's
                prepared = true; gpuWhy = why
                if (why.isNotEmpty()) System.err.println("send: no GPU painter: $why")
                configure()
            }
        }
    }
    fun chooseBlocks(b: Int) { blocks = b.coerceIn(0, 128); Prefs.blocks = blocks }
    fun chooseFps(v: Int) { fps = v.coerceIn(1, 60); Prefs.fps = fps }
    fun chooseSize(v: Int) { size = v.coerceIn(25, 100); Prefs.size = size }
    fun chooseGap(v: Int) { gap = v.coerceIn(0, 64); Prefs.gap = gap }
    fun chooseRing(v: Int) { ring = if (v in 0..3) v else -1; Prefs.ring = ring; repick() }
    fun chooseCodes(v: Int) { codes = if (v == 2) 2 else 1; Prefs.codes = codes; repick() }
    fun commitFps() {
        if (phase != Phase.On) return
        configure()
        if (p != 0L) Native.presentFps(p, fps)
    }
    fun commitSize() { if (p != 0L) Native.presentSize(p, size / 100f); repick() }
    fun repick() {
        if (phase != Phase.On) return
        if (p == 0L) begin() else configure()
    }

    // ai: the canvas's size (a resize, a window made); sending, a re-pick, or the start once it has one. While presenting
    // ai: the re-pick waits for the area to settle, 300 ms (the web's roomChanged): full screen passes through two or three
    // ai: sizes (the column folded, the manager's first size, the screen's), and a re-pick at each restarted the painters
    // ai: (2026-10-04, on a real screen: about 0.5 s of stalled presents a transition); the presenter meanwhile
    // ai: scales the picture it has into the new area.
    fun areaChanged() {
        val (w, hh) = if (canvas.isDisplayable) canvas.devicePx() else 0 to 0
        areaW = w; areaH = hh
        if (p == 0L) repick() else settle.restart()
    }
    private val settle = Timer(300) { repick() }.apply { isRepeats = false }

    // ai: the canvas's window going: the presenter stopped first (its surface is that window); begun again on the next
    fun canvasGone() {
        if (p != 0L) { Native.presentStop(p); p = 0L }
        canvas.presenting = false
    }

    private fun begin() {
        if (h == 0L || p != 0L || !canvas.isShowing || areaW <= 0 || areaH <= 0) return
        if (!configure()) return
        canvas.presenting = true
        val pr = Native.presentStart(canvas, h, fps, size / 100f, false, "")
        if (pr == 0L) { fail("The code cannot be shown: ${Native.txError()}"); return }
        p = pr
        if (paused) Native.presentPause(p, true)
        if (full) Native.presentFullscreen(p, true)
        timer.start()
    }

    // ai: The format the area holds now, configured where it changed: blocks set by hand win, else the largest the room
    // ai: holds (Pick, the web's pickVersion). false where it was refused (the run then ends with why).
    private fun configure(): Boolean {
        if (h == 0L) return false
        val subch = if (blocks > 0) 8 * blocks else Pick.pick(room, ringIndex)
        val f = Format(Pick.nFor(subch), subch, Pick.span(ringIndex), fps, codes, if (codes > 1) gap else 0,
            when (enc) { "cpu" -> 0; "gpu" -> 1; else -> 2 })
        if (f == fmt) return true
        val threads = min(8, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val err = Native.txConfigure(h, f.n, f.subch, f.span, f.fps, f.codes, f.gap, threads, f.painter, Native.assets)
        if (err.isNotEmpty()) { fail(err); return false }
        fmt = f
        val st = parse(Native.txStats(h))
        System.err.println("send: LIZARD-${f.subch}${if (f.codes > 1) " x ${f.codes}, gap ${f.gap}" else ""}" +
            (if (blocks > 0) " set by hand" else " in a ${room.roundToInt()} px room") + " at ${f.fps} a second, ring ${f.span / 2}, painting on the " +
            (if (st?.str("painter") == "gpu") "GPU (${st.str("device")})" else "CPU, $threads painters") +
            (st?.str("gpuWhy").orEmpty().let { if (it.isNotEmpty()) ", not the GPU: $it" else "" }))
        return true
    }

    // ai: once a second while sending: the sender's and the presenter's stats (each a window of about a second, reset as
    // ai: read); an error in either ends the run with it
    fun poll() {
        if (h == 0L) return
        senderRaw = Native.txStats(h); sender = parse(senderRaw)
        if (p != 0L) { presentRaw = Native.presentStats(p); present = parse(presentRaw) }
        val e = sender?.str("error").orEmpty().ifEmpty { if (p != 0L) present?.str("error").orEmpty() else "" }
        if (e.isNotEmpty()) fail(e)
    }

    // ai: The state line (the web's #state): what is sent, paused, or the error (red; Main.kt colours it)
    val stateLine: String get() = when (val ph = phase) {
        is Phase.Error -> ph.why
        Phase.Preparing -> "Preparing $runName"
        Phase.On -> if (paused) "Paused" else "Sending $runName"
        // ai: idle, the web's "<name>, <size>" (the bytes that go, once the sender says) or the app's "The test stream"
        Phase.Idle -> loadError.ifEmpty { if (test) "The test stream" else file?.let { "${it.name}, ${Fmt.bytes(if (sentBytes > 0) sentBytes else fileSize)}" } ?: "" }
    }

    // ai: The figures (the web's #nums): the rate the pictures actually presented carry (the presenter's fresh pictures a
    // ai: second x codes x blocks a frame x 469 B, so a limit outside the sender shows in the figure; 2026-10-04); the
    // ai: pass, beside it until 2026-10-05, is the readout's.
    val figures: String get() {
        if (phase != Phase.On || paused) return ""
        val f = fmt ?: return ""
        val pr = present ?: return ""
        return Fmt.rate(pr.num("shownFps") * f.codes * (f.subch / 8) * 469 / 1000.0)
    }

    // ai: The readout (the web's #tx, Developer Tools): the format and why, the file, the frame on the surface, a frame's
    // ai: bytes, the pictures shown against the asked rate and the screen's, the refreshes each held, the encoder.
    val readout: String get() {
        val f = fmt ?: return ""
        val sd = sender ?: return ""
        val pr = present
        val b = f.subch / 8
        val l = mutableListOf<String>()
        l += "LIZARD-${f.subch} ($b blocks), picture ${f.n} samples in the ${f.span / 2}-cell ring" +
            if (blocks > 0) ", set by hand" else ", chosen from a ${room.roundToInt()} px room and nothing else"
        if (!runTest) l += "file $fileSize B${sd.num("sentBytes").toLong().let { if (it in 1 until fileSize) ", $it B as sent (zstd)" else "" }}: ${sd.num("chunks").toInt()} chunk${if (sd.num("chunks").toInt() == 1) "" else "s"}, " +
            "${sd.num("lap").toInt()} data blocks a pass, BLAKE3 ${sd.str("root").take(16)}..., header in the light" +
            (sd.num("pass").let { p -> if (p >= 1) "\npass ${floor(p).toInt()}" + (sd.num("offeredKBs").let { k -> if (k > 0 && sd.num("lap") > 0) ", ${Fmt.f1(sd.num("lap") / (k * 1000 / 469))} s a pass" else "" }) else "" })
        l += "frame ${sd.num("width").toInt()} x ${sd.num("side").toInt()} px" + (if (f.codes > 1) ", ${f.codes} side by side ${f.gap} modules apart" else "") +
            (pr?.let { " on a ${it.str("surface").replace("x", " x ")} px surface" } ?: "")
        l += "${if (f.codes > 1) "${f.codes} x " else ""}$b x 469 B = ${f.codes * b * 469} B per frame"
        if (pr != null) {
            l += "shown ${Fmt.f1(pr.num("shownFps"))}/s of ${f.fps} asked, ${Fmt.f1(pr.num("presentsPerSec"))} presents a second, the screen " +
                "${Fmt.f1(pr.num("hz"))} Hz${if (pr.num("modeHz") > 0) " (its mode ${Fmt.f1(pr.num("modeHz"))})" else ""}, times from ${if (pr.flag("presentWait")) "present_wait" else "the acquires"}"
            val held = (pr["held"] as? JsonObject)?.entries?.sortedBy { it.key.toIntOrNull() ?: Int.MAX_VALUE }
                ?.joinToString(", ") { "${it.key}: ${(it.value as? JsonPrimitive)?.contentOrNull ?: "?"}" }.orEmpty()
            l += "pictures by the refreshes each held: ${held.ifEmpty { "none" }}; missed ${pr.num("missed").toInt()}, behind the painter ${pr.num("behind").toInt()}"
            l += "presenter: Vulkan on ${pr.str("device")}${if (pr.flag("paused")) ", paused" else ""}"
        }
        l += (if (sd.str("painter") == "gpu") "encoder: GPU, ${sd.str("device")}, ${Fmt.f1(sd.num("paintMs"))} ms a frame"
            else "encoder: CPU, ${sd.num("painters").toInt()} painters, ${Fmt.f1(sd.num("paintMs"))} ms a frame on one" +
                (if (enc != "cpu" && sd.str("gpuWhy").ifEmpty { gpuWhy }.isNotEmpty()) " (not the GPU: ${sd.str("gpuWhy").ifEmpty { gpuWhy }})" else "")) +
            ", ${sd.num("ahead").toInt()} painted ahead"
        return l.joinToString("\n")
    }

    // ai: the build's time for About (build.gradle.kts buildInfo), as the web's: "Build 2026-10-04 12:00 UTC"
    val about: String by lazy {
        val t = javaClass.getResourceAsStream("/lizard-build.txt")?.bufferedReader()?.readText()?.trim().orEmpty()
        if (t.isEmpty()) "" else "Build " + t.replace("T", " ").replace(Regex(":\\d\\d(\\.\\d+)?Z$"), " UTC")
    }
}

private val json = Json { ignoreUnknownKeys = true }
fun parse(s: String): JsonObject? = runCatching { json.parseToJsonElement(s).jsonObject }.getOrNull()
fun JsonObject.num(k: String) = (this[k] as? JsonPrimitive)?.doubleOrNull ?: 0.0
fun JsonObject.str(k: String) = (this[k] as? JsonPrimitive)?.contentOrNull ?: ""
fun JsonObject.flag(k: String) = (this[k] as? JsonPrimitive)?.booleanOrNull ?: false
