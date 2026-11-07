package dev.lizard.desktop

import dev.lizard.pick.Pick
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import dev.lizard.readout.Readout
import dev.lizard.tar.Bundle
import dev.lizard.tar.TAR_TYPE
import dev.lizard.tar.TarEntry
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import java.awt.Canvas
import java.awt.datatransfer.DataFlavor
import java.awt.dnd.DnDConstants
import java.awt.dnd.DropTarget
import java.awt.dnd.DropTargetAdapter
import java.awt.dnd.DropTargetDropEvent
import javax.swing.SwingUtilities
import java.awt.EventQueue
import java.awt.Graphics
import java.awt.event.ComponentAdapter
import java.awt.event.ComponentEvent
import java.awt.event.HierarchyEvent
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.io.File
import java.nio.file.Files
import javax.swing.Timer
import kotlin.concurrent.thread
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asComposeImageBitmap
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import dev.lizard.ui.*

// ai: The code area's window while codes play: a heavyweight AWT canvas, a native window of its own inside the Compose
// ai: window (Screen.kt's SwingPanel), on which the presenter's swapchain draws. In the window only while a presenter
// ai: runs (`want`, 2026-11-02, as the phone's SurfaceView): idle, the area is Compose's, the file upload or the test
// ai: stream's note, under no window of its own (that look painted here by AWT went, its every resize a frame of bare
// ai: fill before the paint, a blink of the test stream's note on each step of a dragged edge; a hidden canvas would not
// ai: do, its SwingPanel's container painting AWT's grey over Compose's area). White, nothing of AWT painted on it;
// ai: never in the control window while a monitor of its own shows the code (SendState.lease). Before its window goes
// ai: (removeNotify: a dispose, a window made again) it stops the presenter on it. A click on it does nothing; files and
// ai: folders dropped on it are taken (a run stops first).
class CodeCanvas(private val s: SendState) : Canvas() {
    // ai: how near the left edge the pointer shows full screen's rail, the window's units
    private val EDGE_NEAR = 24
    // ai: wanted in the window (a presenter is to draw into it; Screen.kt composes its SwingPanel while it is)
    var shown by mutableStateOf(false)
        private set

    init {
        background = java.awt.Color.WHITE
        isFocusable = false
        addComponentListener(object : ComponentAdapter() {
            override fun componentResized(e: ComponentEvent) = s.canvasMoved()
            override fun componentMoved(e: ComponentEvent) = s.canvasMoved()
        })
        if (Native.mac) addHierarchyBoundsListener(object : java.awt.event.HierarchyBoundsAdapter() {
            override fun ancestorMoved(e: HierarchyEvent) = s.canvasMoved()
            override fun ancestorResized(e: HierarchyEvent) = s.canvasMoved()
        })
        // ai: full screen's rail near the left edge (the column's edge is Compose's: the canvas begins past its strip,
        // ai: SendHost.edgeRoom)
        addMouseMotionListener(object : MouseAdapter() {
            override fun mouseMoved(e: MouseEvent) {
                if (s.full && !s.leased && e.x < EDGE_NEAR && !s.edgeRail) s.edgeRail = true
            }
        })
        // ai: Files and folders dropped are added (a run stops first, as the web's does; refused while sending until
        // ai: 2026-10-23); the window's column and the idle area take them too (Screen.kt)
        dropTarget = DropTarget(this, object : DropTargetAdapter() {
            override fun dragEnter(e: java.awt.dnd.DropTargetDragEvent) = s.dragging(true)
            override fun dragExit(e: java.awt.dnd.DropTargetEvent) = s.dragging(false)
            override fun drop(e: DropTargetDropEvent) {
                s.dragging(false)
                e.acceptDrop(DnDConstants.ACTION_COPY)
                e.dropComplete(s.dropped(e.transferable))
            }
        })
    }

    // ai: put in the window for a presenter (its window made at addNotify, which begins it: SendState.canvasMade), taken
    // ai: out at the presenter's end
    fun want(on: Boolean) { shown = on }

    // ai: its X background the area's colour (SendState.codeFill: white round the presenter's code, which clears to
    // ai: white), so what X fills on a map or a resize before anything draws is that (AWT sets the desktop theme's grey:
    // ai: a dark frame where the canvas came, 2026-11-02)
    private fun look() { if (s.loadError.isEmpty() && isDisplayable) runCatching { Native.windowBackground(this, s.codeFill.toArgb() and 0xFFFFFF) } }

    // ai: nothing painted by AWT: the presenter's pictures are the canvas's (Canvas's own paint clears it)
    override fun update(g: Graphics) {}
    override fun paint(g: Graphics) {}

    // ai: its window made (a presenter wanted): its X background the area's colour (Native.windowBackground), then the
    // ai: presenter begun on it
    override fun addNotify() {
        super.addNotify()
        look()
        EventQueue.invokeLater { s.canvasMade() }
    }
    override fun removeNotify() { s.canvasGone(); super.removeNotify() }

    // ai: macOS: where the presenter's layer goes (native/macos.h Place): the canvas's bounds in points from the top left
    // ai: of its window's content (the frame's insets are its title bar), and the screen's scale
    fun place(): FloatArray {
        val win = SwingUtilities.getWindowAncestor(this) ?: return FloatArray(5)
        val at = SwingUtilities.convertPoint(this, 0, 0, win)
        val ins = win.insets
        val k = graphicsConfiguration?.defaultTransform?.scaleX ?: 1.0
        return floatArrayOf((at.x - ins.left).toFloat(), (at.y - ins.top).toFloat(), width.toFloat(), height.toFloat(), k.toFloat())
    }
}

// ai: The run (the web's send.mjs, lizard-android/.../Send.kt SendState): the settings, kept (Prefs), the sender and
// ai: the presenter, and what the column shows. Start makes the sender (txCreate; txPrepare off the UI thread whatever
// ai: the encoder: the presenter shows the code from the painter's device), then, once the code area has a size and is
// ai: shown, picks the format from its room, configures, and starts the presenter on the canvas. A change of the area's
// ai: size, Size, Spacing, Margin, Ring, Codes or Blocks re-picks, and configures where the format changed (a configure
// ai: drops the frames painted ahead; the transfer goes on, as the web's re-pick); FPS configures (the word states it)
// ai: and tells the presenter. Every Native call here is on the UI thread but txCreate and txPrepare; the native side
// ai: holds one lock for configure and the presenter's take.
class SendState {
    sealed interface Phase { data object Idle : Phase; data object Preparing : Phase; data object On : Phase; data class Error(val why: String) : Phase }
    // ai: what is painted: the format configured (its picture the sender's own, lizard_n_for, in its stats)
    data class Format(val subch: Int, val span: Int, val fps: Int, val codes: Int, val gap: Int, val painter: Int)

    // ai: What Start sends (2026-10-16): one file as itself, or several files or a folder as one tar
    // ai: (../liblizard/kotlin/Tar.kt Bundle, the web sender's rule), made from the selection (items); `reading` while
    // ai: picks are walked or the bundle is made
    var choice by mutableStateOf<Bundle?>(null)
    private var singleFile: File? = null   // ai: the one file the choice is, where it is one
    var reading by mutableStateOf(false)
    // ai: The selection (2026-10-23; until then a pick replaced the last): items added by each pick, drop or path handed
    // ai: in (Main.kt), in order, each a file or a folder under its own name, files and folders mixed. An item already
    // ai: listed (its canonical path) is not added again. Links: one picked is followed; those inside a folder are
    // ai: skipped (never followed, so a loop of them cannot make a bundle without end) and counted on its row. A folder
    // ai: that cannot be listed is an error, the list left as it was. Every change stops a run first, then the bundle is
    // ai: made off the UI thread and its transfer made again (make), as the web's does.
    class Item(val key: String, val file: File, val name: String, val dir: Boolean, val files: Int, val size: Long, val links: Int, val entries: List<TarEntry>)
    val items = mutableStateListOf<Item>()
    private var walks = 0           // ai: picks being walked
    private var bundling = 0        // ai: the newest bundle asked for; bundleNow while it is being made
    private var bundleNow = false
    private fun readingNow() { reading = walks > 0 || bundleNow }
    // ai: a drag over the window (the canvas's drop target, Screen.kt's for the rest): the idle area's outline
    var dragOver by mutableStateOf(false)
        private set
    fun dragging(on: Boolean) { dragOver = on }
    // ai: what a drop hands over: its files and folders added (the test stream off); false where it holds none
    fun dropped(t: java.awt.datatransfer.Transferable): Boolean {
        @Suppress("UNCHECKED_CAST")
        val files = runCatching { t.getTransferData(DataFlavor.javaFileListFlavor) as List<File> }.getOrNull().orEmpty()
        if (files.isEmpty()) return false
        if (test) chooseTest(false)
        pick(files)
        return true
    }
    var fileSize by mutableStateOf(0L)
    // ai: the file's bytes as they go, every chunk's zstd frame or its own (the sender's stats, 2026-10-05); 0 before a send
    val sentBytes: Long get() = sender?.num("sentBytes")?.toLong() ?: 0L
    // ai: The transfer made at Start (2026-11-01; at the pick and again at each stop from 2026-10-19, every chunk
    // ai: compressed for picks never sent): the bundle's tar written and every chunk compressed (txCreate) on a thread of
    // ai: its own, the line "Preparing" meanwhile. compressed: its bytes as they go, 0 until a run makes it, kept through
    // ai: the run and after it until the choice changes. A new pick or the window's close lets a made one go.
    var compressed by mutableStateOf(0L)
    private var made = 0L                 // ai: the transfer made for the choice, not yet sending (0: none)
    private var madeTar: File? = null     // ai: its tar, where the choice is a bundle
    private var making = 0                // ai: the make under way's number: a newer make or pick orphans it
    private var makingNow = false
    private var startWhenMade = false     // ai: Start pressed while the transfer was being made
    private var closing = false
    var test by mutableStateOf(Prefs.payload == "test")
    var enc by mutableStateOf(Prefs.enc.let { if (it == "gpu" || it == "cpu") it else "auto" })
    // ai: the painter the sender runs on now, "gpu" or "cpu" ("" before a send): the Encoder chip auto marks (2026-10-05)
    val painterNow: String get() = sender?.str("painter") ?: ""
    // ai: 0 auto, else a version 1 to 64 (16 sub-channels a step, Pick.STEP)
    var blocks by mutableStateOf(Prefs.blocks.let { if (it in Pick.VERSION_MIN..Pick.VERSION_MAX) it else 0 })
    var fps by mutableStateOf(if (Prefs.vsync) Pick.fpsOf(Prefs.fps) else Prefs.fps.coerceIn(1, 60))
    // ai: VSync (2026-10-23, on by default): while the code is shown its monitor holds a refresh the rate divides, the
    // ai: nearest 60 Hz, where its own does not (a 165 Hz laptop at 60: pictures of 3, 3, 2 and 3 refreshes), and is put
    // ai: back at the stop (presenter.h setVsync); off, the monitor is left as it is. FPS is Pick.FPS's chips under VSync,
    // ai: a 1 to 60 slider without it; VSync turned on takes a slider's rate to the nearest chip.
    var vsync by mutableStateOf(Prefs.vsync)
    var size by mutableStateOf(Prefs.size.coerceIn(25, 100))
    var spacing by mutableStateOf(Prefs.spacing.coerceIn(Room.SPACING_MIN, Room.SPACING_MAX))
    var margin by mutableStateOf(Prefs.margin.coerceIn(Pick.QUIET, Room.MARGIN_MAX))
    var ring by mutableStateOf(Prefs.ring.let { if (it in Pick.RINGS.indices) it else -1 })
    var codes by mutableStateOf(if (Prefs.codes == 2) 2 else 1)
    var settingsOpen by mutableStateOf(Prefs.settingsOpen)
    var toolsOpen by mutableStateOf(Prefs.toolsOpen)
    var experimentalOpen by mutableStateOf(Prefs.experimentalOpen)
    var collapsed by mutableStateOf(Prefs.collapsed)

    var phase by mutableStateOf<Phase>(Phase.Idle)
    var paused by mutableStateOf(false)
    // ai: Full screen (the Fullscreen button, F or F11; Escape, F, F11 or the rail's collapser leaves), as other apps have it
    // ai: (2026-10-23): the window manager's own (Main.kt fullscreen), the code alone in it as the web's and the app's;
    // ai: the manager's word followed (fullFromManager). Until then the column folded to the rail, which a toggle opened
    // ai: there, and the presenter bypassed the compositor.
    var full by mutableStateOf(false)
    var fullscreen: ((Boolean) -> Unit)? = null
    // ai: Full screen on the display itself (2026-10-31, Exclusive fullscreen on): a Fullscreen press while the codes
    // ai: play (sending, not paused) takes the monitor under the window's centre from the desktop (Native.presentStart's
    // ai: direct; X11 through VK_EXT_acquire_xlib_display) and shows the code alone on it, presented FIFO at the
    // ai: display's own refresh. With another monitor the window, moved there by the window manager, is the control
    // ai: (Pause, Stop, Fullscreen, its keys; the code area plain); on the only monitor the presenter holds the keyboard
    // ai: and the pointer and any key or click gives it back, as Stop does. The window is then put back where it was
    // ai: (keepPlace before, placeBack after, App.kt). Where no monitor is lent (Windows, macOS, Wayland, a driver
    // ai: without the extensions, a press while nothing plays), the window manager's full screen as before.
    // ai: an experimental setting, off unless chosen (Prefs.exclusiveFull; X11 alone offers the lease): off, Fullscreen is
    // ai: the window manager's full screen whatever the system
    var exclusiveFull by mutableStateOf(Prefs.exclusiveFull)
    var leased by mutableStateOf(false)
    // ai: the taken monitor's name (its EDID's, else the output's: the stats' directName), the control window's line
    var leasedOn by mutableStateOf("")
        private set
    var keepPlace: (() -> Unit)? = null
    var placeBack: (() -> Unit)? = null
    // ai: full screen's rail shown (Screen.kt EdgeRail): the pointer came within EDGE_NEAR of the code area's left edge
    var edgeRail by mutableStateOf(false)
    // ai: the window maximized, as a title bar's double click does (Main.kt sets it; the test hook's `max` step)
    var maximize: (() -> Unit)? = null
    // ai: the window brought to the front, as paths handed in by a later launch arrive (Main.kt)
    var raise: (() -> Unit)? = null
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
    // ai: while the monitor is taken: the presenter's word read five times a second, so a key or a click that gave it
    // ai: back brings the window back at once
    private val leaseTimer = Timer(200) { if (leased && p != 0L && parse(Native.presentStats(p))?.str("direct").isNullOrEmpty()) giveBack() }

    val presenting get() = p != 0L
    // ai: the code area's colour at its left edge (the column's margin beside it takes it, SendHost.codeFill; the canvas's
    // ai: X background): the soft fill under the test stream's note and while a monitor of its own shows the code, white
    // ai: under the file upload, round the presenter's code (it clears to white) and round a paused run's still
    val codeFill: Color get() = if (leased || (test && !canvas.shown && stamp == null)) Soft else Bg
    val sending get() = phase == Phase.On || phase == Phase.Preparing
    val canStart get() = loadError.isEmpty() && !sending && (test || (choice != null && !reading))
    val railShown get() = collapsed
    val ringIndex get() = if (ring in Pick.RINGS.indices) ring else Pick.RING_DEFAULT
    // ai: a symbol's room in the area, device px, the Size slider's share of it (send.mjs room)
    val room get() = Room.room(areaW, areaH, codes, ringIndex, spacing, margin) * size / 100.0
    // ai: the margin past the quiet zone the presenter keeps at the area's edges, for the ring configured
    private val edge get() = edgeShare
    // ai: the light the presenter keeps past the frame at each edge, a share of its height (presenter.h PresentOptions::edge)
    val edgeShare get() = Room.edgeShare(fmt?.span ?: Pick.span(ringIndex), margin).toFloat()
    // ai: what auto takes in the area now (0 while it has no size): the configured format's while sending
    val autoBlocks get() = fmt?.takeIf { blocks == 0 }?.let { it.subch / Pick.STEP } ?: if (areaW > 0 && areaH > 0) Pick.pick(room, ringIndex) / Pick.STEP else 0
    // ai: what the code offers at most, KB/s: codes x blocks a frame x 472 B x the pictures a second it can show, the
    // ai: asked rate or the measured refresh where lower, so a limit outside the sender shows in the figure
    // ai: (2026-10-04); null before a configure
    val capacityKBs: Double? get() {
        val f = fmt ?: return null
        val hz = present?.num("hz") ?: 0.0
        return f.codes * sentBlocks * Pick.PAYLOAD.toDouble() * (if (hz > 0) min(f.fps.toDouble(), hz) else f.fps.toDouble()) / 1000
    }
    // ai: the blocks a code of the painted format carries, from the sender's stats, the configured format's before the
    // ai: first stats; its profile, "" none
    val sentBlocks: Int get() = sender?.num("blocks")?.toInt()?.takeIf { it > 0 } ?: fmt?.let { Native.blocksFor(it.subch) } ?: 0
    val tiers: String get() = sender?.str("tiers").orEmpty()

    // ai: The chooser (Chooser.kt): several files, or several folders, over the sender's window
    // ai: the window (App.kt sets it): the choosers' owner (the canvas has a window only while codes play)
    var frame: java.awt.Component? = null
    fun chooseFile() = frame?.let { Chooser.choose(it, "Choose files", false, true) { f -> pick(f) } }
    fun chooseFolder() = frame?.let { Chooser.choose(it, "Choose folders", true, true) { f -> pick(f) } }

    // ai: Files and folders added (picked, dropped, handed in), walked off the UI thread
    fun pick(files: List<File>) {
        if (sending) stop()
        if (phase is Phase.Error) phase = Phase.Idle
        walks++; readingNow()
        thread(name = "lizard-send-pick", isDaemon = true) {
            val got = runCatching { files.map { itemOf(it) } }
            EventQueue.invokeLater {
                walks--; readingNow()
                got.onSuccess { list ->
                    val have = items.mapTo(HashSet()) { it.key }
                    val fresh = list.filter { have.add(it.key) }
                    if (fresh.isNotEmpty()) { items.addAll(fresh); changed() }
                }.onFailure { phase = Phase.Error("The files could not be read: ${it.message}") }
            }
        }
    }
    fun remove(key: String) { items.removeAll { it.key == key }; changed() }
    fun clearPicks() { items.clear(); changed() }
    private fun itemOf(f: File): Item {
        val top = f.canonicalFile
        var links = 0
        val out = ArrayList<TarEntry>()
        fun walk(x: File, rel: String) {
            if (x.isDirectory) {
                out.add(TarEntry(rel, mtimeSec = x.lastModified() / 1000))
                val kids = x.listFiles() ?: throw java.io.IOException("$rel could not be read")
                for (k in kids.sortedBy { it.name }) if (Files.isSymbolicLink(k.toPath())) links++ else walk(k, "$rel/${k.name}")
            } else if (x.isFile) out.add(TarEntry(rel, x.length(), x.lastModified() / 1000) { x.inputStream() })
        }
        walk(top, f.name)
        if (out.isEmpty()) throw java.io.IOException("${f.name} is not a file or a folder")
        return Item(top.path, top, f.name, out[0].dir, out.count { !it.dir }, out.filter { !it.dir }.sumOf { it.size }, links, out)
    }
    // ai: the selection changed: the run stopped, the made transfer let go, the bundle made again off the UI thread (its
    // ai: transfer waits for Start)
    private fun changed() {
        if (sending) stop()
        dropMade()
        if (phase is Phase.Error) phase = Phase.Idle
        choice = null; singleFile = null; fileSize = 0; compressed = 0
        val list = items.toList()
        val my = ++bundling
        bundleNow = list.isNotEmpty(); readingNow()
        if (list.isEmpty()) return
        thread(name = "lizard-send-bundle", isDaemon = true) {
            val b = Bundle(list.flatMap { it.entries })
            EventQueue.invokeLater {
                if (my != bundling) return@invokeLater
                bundleNow = false; readingNow()
                choice = b; singleFile = if (b.single != null) list[0].file else null; fileSize = b.size
            }
        }
    }
    // ai: a bundle's tar, written for the sender to map (txCreate maps a file) and deleted when the run ends
    private var tarFile: File? = null
    private fun dropTar() { tarFile?.delete(); tarFile = null }

    // ai: The choice's transfer made off the UI thread: its tar written where it is a bundle, txCreate (each chunk
    // ai: compressed, its fountain set up), its bytes as sent read from the stats. A failure is the state line's at once.
    private fun make() {
        val b = choice ?: return
        dropMade()
        val my = making
        makingNow = true
        val one = if (b.single == null) null else singleFile
        val tar = if (one != null) null else File(Native.cache, "send-${ProcessHandle.current().pid()}-$my.tar")
        val path = one?.path ?: tar!!.path
        val type = if (one == null) TAR_TYPE else runCatching { Files.probeContentType(one.toPath()) }.getOrNull() ?: ""
        thread(name = "lizard-send-make", isDaemon = true) {
            val wrote = if (tar == null) null else runCatching {
                tar.parentFile.mkdirs(); tar.deleteOnExit()
                tar.outputStream().buffered(1 shl 20).use { b.write(it) }
            }.exceptionOrNull()
            val hnd = if (wrote != null) 0L else Native.txCreate(path, b.name, type)
            val err = if (wrote != null) "The files could not be read: ${wrote.message}" else if (hnd == 0L) "This file cannot be sent: ${Native.txError()}" else ""
            val sent = if (hnd != 0L) parse(Native.txStats(hnd))?.num("sentBytes")?.toLong() ?: 0L else 0L
            EventQueue.invokeLater {
                if (my != making || closing) { if (hnd != 0L) Native.txDestroy(hnd); tar?.delete(); return@invokeLater }   // ai: let go meanwhile
                makingNow = false
                if (hnd == 0L) { tar?.delete(); startWhenMade = false; phase = Phase.Error(err); return@invokeLater }
                made = hnd; madeTar = tar; compressed = sent
                if (startWhenMade) { startWhenMade = false; go() }
            }
        }
    }
    // ai: the made transfer let go, and any make under way orphaned
    private fun dropMade() {
        making++; makingNow = false; startWhenMade = false
        if (made != 0L) { Native.txDestroy(made); made = 0L }
        madeTar?.delete(); madeTar = null
    }

    // ai: Start (the column's, the rail's play, the hook's); no tip before it since 2026-11-03
    fun start() {
        if (!canStart) return
        runTest = test
        runName = if (test) "" else choice!!.name
        dropTar()
        phase = Phase.Preparing; paused = false
        // ai: the file's transfer still being made (or its make let go): Start waits for it
        if (!test && made == 0L) { if (!makingNow) make(); startWhenMade = true; return }
        go()
    }
    // ai: Sending begins: the made transfer taken (the test stream's made here), the GPU painter's device and pipelines
    // ai: made off the UI thread (txPrepare, whatever the encoder: the presenter shows the code from that device,
    // ai: 2026-10-07), then the pick and the presenter once the canvas has a size
    private fun go() {
        val pre = if (runTest) 0L else made
        if (!runTest) { made = 0L; tarFile = madeTar; madeTar = null }
        val my = ++gen
        thread(name = "lizard-send-prep", isDaemon = true) {
            val hnd = if (pre != 0L) pre else Native.txCreate("", "", "")
            val err = if (hnd == 0L) Native.txError() else ""
            // ai: the GPU painter's device and pipelines made here, off the UI thread that configures
            val why = if (hnd != 0L) Native.txPrepare(hnd, Native.assets) else ""
            EventQueue.invokeLater {
                if (my != gen) { if (hnd != 0L) Native.txDestroy(hnd); return@invokeLater }   // ai: stopped meanwhile
                if (hnd == 0L) { phase = Phase.Error("This file cannot be sent: $err"); return@invokeLater }
                h = hnd; gpuWhy = why; prepared = true; fmt = null; phase = Phase.On
                if (why.isNotEmpty()) System.err.println("send: no GPU painter: $why")
                areaNow()
            }
        }
    }

    fun stop() {
        gen++
        timer.stop()
        settle.stop()
        if (p != 0L) { Native.presentStop(p); p = 0L }
        if (leased) leaseEnded()
        canvas.want(false)
        stamp = null
        if (h != 0L && h != preparing) Native.txDestroy(h)
        h = 0L
        dropTar()
        startWhenMade = false
        if (sending) phase = Phase.Idle
        paused = false; fmt = null; present = null; sender = null
    }
    // ai: the window closing: the run stopped and the made transfer let go
    fun close() { closing = true; stop(); dropMade() }
    // ai: Send left for another screen (App.go, 2026-10-25; the Android app's go()): full screen left while its canvas
    // ai: still has a window, the run stopped (the presenter's canvas leaves the window with the
    // ai: screen); the made transfer kept for a return
    fun leave() {
        if (full) toggleFull()
        edgeRail = false
        if (sending) stop()
    }

    private fun fail(why: String) { stop(); phase = Phase.Error(why) }

    // ai: play and pause as a video player's (the rail's button, Space and K): Start where nothing runs, else
    // ai: Pause or Resume
    fun playPause() { if (phase == Phase.On) pause(!paused) else if (!sending) start() }
    // ai: Paused: the last picture stays; a receiver holds what it has. In the window (2026-11-02) it becomes a still:
    // ai: the picture on screen read back (`stamp`), the presenter stopped and the canvas out of the window, so the area
    // ai: is Compose's (DesktopSend.Stamped) and a resize moves nothing native; Resume makes both again, the format picked
    // ai: for the area as it is then. On a monitor of its own the presenter holds it there ("Paused on").
    class Stamp(val w: Int, val h: Int, val image: ImageBitmap)
    var stamp by mutableStateOf<Stamp?>(null)
        private set
    fun pause(on: Boolean) {
        if (phase != Phase.On) return
        paused = on
        if (leased) {
            if (p != 0L) Native.presentPause(p, on)
            if (!on) takeOver()
            return
        }
        if (on) {
            if (p != 0L) {
                Native.presentPause(p, true)
                stamp = snapshot(p)
                Native.presentStop(p); p = 0L
            }
            canvas.want(false)
        } else {
            begin()
        }
    }
    private fun snapshot(p: Long): Stamp? {
        val wh = IntArray(2)
        val px = Native.presentSnapshot(p, wh) ?: return null
        val (w, h) = wh[0] to wh[1]
        if (w <= 0 || h <= 0 || px.size < w * h) return null
        val bm = org.jetbrains.skia.Bitmap()
        if (!bm.installPixels(org.jetbrains.skia.ImageInfo(w, h, org.jetbrains.skia.ColorType.GRAY_8, org.jetbrains.skia.ColorAlphaType.OPAQUE), px, w)) return null
        bm.setImmutable()
        return Stamp(w, h, bm.asComposeImageBitmap())
    }
    // ai: The code area resized by hand while the codes play in the window (the column's edge dragged, the window
    // ai: resized; 2026-11-02): paused, a still until Resume. Not a full screen's own sizes (toggleFull, the manager's
    // ai: word, a monitor given back: `quietUntil`), nor a fold of the column to the rail (a re-pick, as before).
    private var quietUntil = 0L
    private fun quiet() { quietUntil = System.nanoTime() + 1_500_000_000L }
    fun resizing() { if (phase == Phase.On && !paused && !leased && !full && System.nanoTime() > quietUntil) pause(true) }

    fun toggleFull() {
        edgeRail = false
        quiet()
        if (leased) { giveBack(); return }
        full = !full
        if (full && exclusiveFull && phase == Phase.On && !paused && p != 0L && lease()) return
        fullscreen?.invoke(full)
    }
    // ai: Exclusive fullscreen from the window manager's full screen: the codes began playing there (Start, Resume) or
    // ai: the setting was turned on while they play. That full screen is left first, so keepPlace keeps the window's
    // ai: own place, and the monitor taken once the manager has given the window back (takeOverSoon); where no monitor
    // ai: is lent the window goes back to the manager's full screen, and no take-over is tried again this session
    private var leaseRefused = false
    private fun takeOver() {
        if (!full || leased || leaseRefused || !exclusiveFull || phase != Phase.On || paused || p == 0L) return
        fullscreen?.invoke(false)
        takeOverSoon.restart()
    }
    private val takeOverSoon = Timer(400) {
        if (leased || !exclusiveFull || phase != Phase.On || paused || p == 0L) return@Timer
        full = true
        if (!lease()) { leaseRefused = true; fullscreen?.invoke(true) }
    }.apply { isRepeats = false }
    // ai: the presenter begun again on the monitor under the window, the display or nothing (presenter.h directOnly,
    // ai: 2026-11-02): handed the app's window, which only names the monitor and takes the grab, so the control window
    // ai: has no canvas and its area is Compose's (DesktopSend.LeasedNote). true where the monitor was taken; else the
    // ai: codes go on in the window as before, and no take-over is tried again this session.
    private fun lease(): Boolean {
        val win = frame ?: return false
        keepPlace?.invoke()
        Native.presentStop(p); p = 0L
        val pr = Native.presentStart(win, h, fps, size / 100f, edge, vsync, true, FloatArray(5))
        if (pr == 0L) {
            System.err.println("send: no monitor taken: ${Native.txError()}")
            leaseRefused = true
            begin()
            return false
        }
        p = pr
        if (paused) Native.presentPause(p, true)
        val st = parse(Native.presentStats(p))
        leased = true
        leasedOn = st?.str("directName").orEmpty().ifEmpty { st?.str("direct").orEmpty() }
        canvas.want(false)
        // ai: the format for the monitor taken: its pixels the area (an auto pick, Size, Margin and Spacing measured on it),
        // ai: held while leased; each setting changed in the control window re-picks on it
        st?.str("directPx")?.split("x")?.mapNotNull { it.toIntOrNull() }?.takeIf { it.size == 2 && it[0] > 0 && it[1] > 0 }
            ?.let { areaW = it[0]; areaH = it[1] }
        leaseTimer.start(); repick()
        return true
    }
    // ai: the monitor given back by hand (Fullscreen, Escape, a key or a click to the presenter): that presenter stopped
    // ai: (it has no window to go on in), the codes on in the window, or, paused, the picture there a still
    private fun giveBack() {
        if (p != 0L) { if (paused) stamp = snapshot(p); Native.presentStop(p); p = 0L }
        leaseEnded()
        if (sending && !paused) begin() else canvas.want(false)
    }
    // ai: the monitor given back (a key or a click to the presenter, Stop, the window closing): full screen left and
    // ai: the window where it was before
    private fun leaseEnded() {
        quiet()
        leaseTimer.stop()
        leased = false
        full = false
        placeBack?.invoke()
        // ai: the area's size followed again (the window's presenter, begun after, picks from it)
        if (seenW > 0 && seenH > 0) { areaW = seenW; areaH = seenH }
    }
    // ai: the manager's own word on the window (its shortcut, a leave of its own, a request of ours taken): followed,
    // ai: nothing asked
    fun fullFromManager(on: Boolean) { if (!leased && on != full) { full = on; edgeRail = false; quiet() } }

    fun toggleRail() { collapsed = !collapsed; Prefs.collapsed = collapsed }
    fun toggleSettings() { settingsOpen = !settingsOpen; Prefs.settingsOpen = settingsOpen }
    fun toggleTools() { toolsOpen = !toolsOpen; Prefs.toolsOpen = toolsOpen }
    fun toggleExperimental() { experimentalOpen = !experimentalOpen; Prefs.experimentalOpen = experimentalOpen }

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
    fun chooseBlocks(b: Int) { blocks = if (b in Pick.VERSION_MIN..Pick.VERSION_MAX) b else 0; Prefs.blocks = blocks }
    fun chooseFps(v: Int) { fps = v.coerceIn(1, 60); Prefs.fps = fps }
    fun chooseVsync(on: Boolean) {
        vsync = on; Prefs.vsync = on
        if (on && fps !in Pick.FPS) { chooseFps(Pick.fpsOf(fps)); commitFps() }
        if (p != 0L) Native.presentVsync(p, on)
    }
    fun chooseSize(v: Int) { size = v.coerceIn(25, 100); Prefs.size = size }
    fun chooseSpacing(v: Int) { spacing = v.coerceIn(Room.SPACING_MIN, Room.SPACING_MAX); Prefs.spacing = spacing }
    fun chooseMargin(v: Int) { margin = v.coerceIn(Pick.QUIET, Room.MARGIN_MAX); Prefs.margin = margin }
    fun chooseRing(v: Int) { ring = if (v in Pick.RINGS.indices) v else -1; Prefs.ring = ring; repick() }
    fun chooseCodes(v: Int) { codes = if (v == 2) 2 else 1; Prefs.codes = codes; repick() }
    // ai: turned on in full screen while the codes play: the monitor taken (takeOver); turned off while it is: given back
    fun chooseExclusiveFull(on: Boolean) {
        exclusiveFull = on; Prefs.exclusiveFull = on
        if (on) takeOver() else if (leased) toggleFull()
    }
    fun commitFps() {
        if (phase != Phase.On) return
        configure()
        if (p != 0L) Native.presentFps(p, fps)
    }
    fun commitSize() = repick()
    fun repick() {
        if (phase != Phase.On) return
        if (p == 0L) begin() else if (configure()) Native.presentSize(p, size / 100f, edge)
    }

    // ai: the code area's size in device px as Compose lays it out (Screen.kt's onSizeChanged: the canvas, hidden while
    // ai: idle, is laid out by no one then); sending, a re-pick, or the start once it has one. While presenting
    // ai: the re-pick waits for the area to settle, 300 ms (the web's roomChanged): full screen passes through two or three
    // ai: sizes (the column folded, the manager's first size, the screen's), and a re-pick at each restarted the painters
    // ai: (2026-10-04, on a real screen: about 0.5 s of stalled presents a transition); the presenter meanwhile
    // ai: scales the picture it has into the new area.
    private var seenW = 0
    private var seenH = 0
    fun areaChanged(w: Int, hh: Int) { seenW = w; seenH = hh; areaNow() }
    private fun areaNow() {
        // ai: while a monitor is taken the area is the monitor's (lease): the control window's is no measure of it (read
        // ai: again at leaseEnded)
        if (leased) return
        // ai: an area not yet laid out keeps the area it had: Blocks' label read "Auto, N" -> "Auto" -> "Auto, N" on each
        // ai: entry to Send (2026-11-02)
        if (seenW <= 0 || seenH <= 0) return
        areaW = seenW; areaH = seenH
        if (p == 0L) repick() else settle.restart()
    }
    private val settle = Timer(300) { repick() }.apply { isRepeats = false }

    // ai: macOS: the presenter's layer follows the canvas (a move or a resize of it or of the window)
    fun canvasMoved() { if (p != 0L && !leased) Native.presentPlace(p, canvas.place()) }

    // ai: the canvas's window going: the presenter stopped first (its surface is that window); begun again on the next
    // ai: (a leased presenter is not on it: handed the app's window, it goes on; stopping it here gave the monitor back
    // ai: the moment the lease took the canvas out, 2026-11-02)
    fun canvasGone() {
        if (p != 0L && !leased) { Native.presentStop(p); p = 0L }
    }
    // ai: the canvas's window made: the presenter begun where one is wanted
    fun canvasMade() { if (phase == Phase.On && p == 0L) repick() }

    // ai: the presenter needs the canvas's window made (wanted first: canvasMade comes back here), not shown: a Start at
    // ai: the launch (the hook's) comes before AWT counts the window as showing. None while paused (a still; Resume).
    private fun begin() {
        if (h == 0L || p != 0L || paused || areaW <= 0 || areaH <= 0) return
        if (!canvas.isDisplayable) { canvas.want(true); return }
        if (!configure()) return
        val pr = Native.presentStart(canvas, h, fps, size / 100f, edge, vsync, false, canvas.place())
        if (pr == 0L) { fail("The code cannot be shown: ${Native.txError()}"); return }
        p = pr
        stamp = null
        if (paused) Native.presentPause(p, true)
        timer.start()
        takeOver()
    }

    // ai: The format the area holds now, configured where it changed: blocks set by hand win, else the largest the room
    // ai: holds (Pick, the web's pickVersion). false where it was refused (the run then ends with why).
    private fun configure(): Boolean {
        if (h == 0L) return false
        val subch = if (blocks > 0) Pick.STEP * blocks else Pick.pick(room, ringIndex)
        val f = Format(subch, Pick.span(ringIndex), fps, codes, if (codes > 1) Room.gapOf(spacing) else 0,
            when (enc) { "cpu" -> 0; "gpu" -> 1; else -> 2 })
        if (f == fmt) return true
        val threads = min(8, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val err = Native.txConfigure(h, f.subch, f.span, f.fps, f.codes, f.gap, threads, f.painter, Native.assets)
        if (err.isNotEmpty()) { fail(err); return false }
        fmt = f
        val st = parse(Native.txStats(h))
        System.err.println("send: LIZARD-${f.subch}${if (f.codes > 1) " x ${f.codes}, spacing $spacing" else ""}, margin $margin" +
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
        // ai: the bytes as they go, known once every chunk is made (the core streams: its first frames go while the
        // ai: chunks are made, 2026-11-06), so the state line takes them from here
        if (compressed == 0L) sender?.num("sentBytes")?.toLong()?.takeIf { it > 0 }?.let { compressed = it }
        if (p != 0L) { presentRaw = Native.presentStats(p); present = parse(presentRaw) }
        val e = sender?.str("error").orEmpty().ifEmpty { if (p != 0L) present?.str("error").orEmpty() else "" }
        if (e.isNotEmpty()) fail(e)
    }

    // ai: what goes, "<compressed> (<raw> raw)" once a run has made the transfer, the size alone before (the web's and the
    // ai: app's the same)
    // ai: "raw" kept with its figure (a no-break space, 2026-10-23: "(47 KB / raw)" broke there)
    private fun sizeText() = if (compressed > 0) "${Readout.bytes(compressed)} (${Readout.bytes(fileSize)}\u00a0raw)" else Readout.bytes(fileSize)

    // ai: The state line (the web's #state): what is sent (paused too: Resume beside Stop tells a pause, no "Paused" line,
    // ai: 2026-11-06), or the error (red; Main.kt colours it). A line naming
    // ai: the file says so in three parts, the name between (2026-10-23), for the shared NameLine to hold its head on one
    // ai: line; any other is all `before`.
    data class StateLine(val before: String, val name: String? = null, val after: String = "")
    val stateParts: StateLine get() = when (val ph = phase) {
        is Phase.Error -> StateLine(ph.why)
        Phase.Preparing -> if (runTest) StateLine("Preparing") else StateLine("Preparing ", runName)
        Phase.On -> if (runTest) StateLine("Sending") else StateLine("Sending ", runName, ", ${sizeText()}")
        // ai: idle, nothing but "Reading the files" or an error: the choice is named on its card (choiceLine, 2026-11-01)
        Phase.Idle -> if (loadError.isNotEmpty()) StateLine(loadError) else if (!test && reading) StateLine("Reading the files") else StateLine("")
    }
    // ai: the choice's name and what follows it, ", N files, " for a bundle and the bytes that go (the web's "<name>,
    // ai: <size>"); null while the pick is walked or the bundle made
    val choiceLine: Pair<String, String>? get() = choice?.takeIf { !reading }?.let {
        it.name to ", ${if (it.single == null) "${it.files} ${if (it.files == 1) "file" else "files"}, " else ""}${sizeText()}"
    }
    val stateLine: String get() = stateParts.let { "${it.before}${it.name ?: ""}${it.after}" }

    // ai: The figures (the web's #nums): the rate the pictures actually presented carry (the presenter's fresh pictures a
    // ai: second x codes x blocks a frame x 472 B, so a limit outside the sender shows in the figure; 2026-10-04); the
    // ai: pass, beside it until 2026-10-05, is the readout's.
    val figures: String get() {
        if (phase != Phase.On || paused) return ""
        val f = fmt ?: return ""
        val pr = present ?: return ""
        return Readout.rate(pr.num("shownFps") * f.codes * sentBlocks * Pick.PAYLOAD / 1000.0)
    }

    // ai: The readout (the web's #tx, Developer Tools): the format and why, the file, the frame on the surface, a frame's
    // ai: bytes, the pictures shown against the asked rate and the screen's, the refreshes each held, the encoder.
    val readout: String get() {
        val f = fmt ?: return ""
        val sd = sender ?: return ""
        val pr = present
        val b = sentBlocks
        val l = mutableListOf<String>()
        l += "LIZARD-${f.subch} ($b blocks${if (tiers.isNotEmpty()) ": $tiers" else ""}), picture ${sd.num("n").toInt()} samples in the ${f.span / 2}-cell ring" +
            if (blocks > 0) ", manual" else ", auto for ${room.roundToInt()} px"
        if (!runTest) l += "file $fileSize B${sd.num("sentBytes").toLong().let { if (it in 1 until fileSize) ", $it B as sent (zstd)" else "" }}: ${sd.num("chunks").toInt()} chunk${if (sd.num("chunks").toInt() == 1) "" else "s"}, " +
            "${sd.num("lap").toInt()} data blocks a pass, BLAKE3 ${sd.str("root").take(16)}..." +
            (sd.num("pass").let { p -> if (p >= 1) "\npass ${floor(p).toInt()}" + (sd.num("offeredKBs").let { k -> if (k > 0 && sd.num("lap") > 0) ", ${Fmt.f1(sd.num("lap") / (k * 1000 / Pick.PAYLOAD))} s a pass" else "" }) else "" })
        l += "frame ${sd.num("width").toInt()} x ${sd.num("side").toInt()} px" + (if (f.codes > 1) ", ${f.codes} side by side $spacing modules apart" else "") + ", margin $margin modules" +
            (pr?.let { " on a ${it.str("surface").replace("x", " x ")} px surface" } ?: "")
        l += "${if (f.codes > 1) "${f.codes} x " else ""}$b x ${Pick.PAYLOAD} B = ${f.codes * b * Pick.PAYLOAD} B per frame"
        if (pr != null) {
            l += "shown ${Fmt.f1(pr.num("shownFps"))}/s of ${f.fps} asked, ${Fmt.f1(pr.num("presentsPerSec"))} presents a second, the screen " +
                "${Fmt.f1(pr.num("hz"))} Hz${if (pr.num("modeHz") > 0) " (its mode ${Fmt.f1(pr.num("modeHz"))})" else ""}" +
                (if (pr.num("modeFrom") > 0) ", held by VSync from ${Fmt.f1(pr.num("modeFrom"))} Hz" else "") + ", times from ${if (pr.flag("presentWait")) "present_wait" else "the acquires"}"
            val held = (pr["held"] as? JsonObject)?.entries?.sortedBy { it.key.toIntOrNull() ?: Int.MAX_VALUE }
                ?.joinToString(", ") { "${it.key}: ${(it.value as? JsonPrimitive)?.contentOrNull ?: "?"}" }.orEmpty()
            l += "refreshes a picture: ${held.ifEmpty { "none" }}; missed ${pr.num("missed").toInt()}, behind ${pr.num("behind").toInt()}"
            l += "presenter: Vulkan on ${pr.str("device")}${if (pr.flag("paused")) ", paused" else ""}"
        }
        l += (if (sd.str("painter") == "gpu") "encoder: GPU, ${sd.str("device")}, ${Fmt.f1(sd.num("paintMs"))} ms a frame"
            else "encoder: CPU, ${sd.num("painters").toInt()} painters, ${Fmt.f1(sd.num("paintMs"))} ms a frame on one" +
                (if (enc != "cpu" && sd.str("gpuWhy").ifEmpty { gpuWhy }.isNotEmpty()) " (not the GPU: ${sd.str("gpuWhy").ifEmpty { gpuWhy }})" else "")) +
            ", ${sd.num("ahead").toInt()} painted ahead"
        return l.joinToString("\n")
    }

    // ai: the build's time for About (build.gradle.kts buildInfo), as the web's: "Build: 2026-10-04 12:00 UTC"
    val about: String by lazy {
        val t = javaClass.getResourceAsStream("/lizard-build.txt")?.bufferedReader()?.readText()?.trim().orEmpty()
        if (t.isEmpty()) "" else "Build: " + t.replace("T", " ").replace(Regex(":\\d\\d(\\.\\d+)?Z$"), " UTC")
    }
}

