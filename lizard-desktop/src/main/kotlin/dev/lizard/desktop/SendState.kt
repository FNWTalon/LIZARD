package dev.lizard.desktop

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import dev.lizard.tar.Bundle
import dev.lizard.tar.TAR_TYPE
import dev.lizard.tar.TarEntry
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
import javax.swing.JFileChooser
import javax.swing.SwingUtilities
import java.awt.Cursor
import java.awt.EventQueue
import java.awt.Graphics
import java.awt.Point
import java.awt.Rectangle
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
// ai: made again), stops the presenter on it. A click on it does nothing in full screen (2026-10-24, owner: "clicking on
// ai: the sender should not remove fullscreen"; Escape, F, F11 and the rail's collapser leave it); the Idle
// ai: (2026-10-07, the web's hint in the middle of its page and the app's code box) it is the file's area: the soft fill, the file icon and "Choose files" over the link "or a folder", or the choice's
// ai: name once one is chosen, or "Test stream"; a click opens the file dialog (the link, the folder chooser), and files
// ai: and folders dropped on it are taken (the test stream off; several, or a folder, one tar since 2026-10-16). AWT
// ai: paints that itself, since nothing of Compose can be drawn over a heavyweight canvas.
class CodeCanvas(private val s: SendState) : Canvas() {
    @Volatile var presenting = false
    // ai: how near the left edge the pointer shows full screen's rail, the window's units
    private val EDGE_NEAR = 24

    init {
        background = java.awt.Color.WHITE
        isFocusable = false
        addComponentListener(object : ComponentAdapter() {
            override fun componentResized(e: ComponentEvent) { s.areaChanged(); s.canvasMoved() }
            override fun componentMoved(e: ComponentEvent) = s.canvasMoved()
        })
        if (Native.mac) addHierarchyBoundsListener(object : java.awt.event.HierarchyBoundsAdapter() {
            override fun ancestorMoved(e: HierarchyEvent) = s.canvasMoved()
            override fun ancestorResized(e: HierarchyEvent) = s.canvasMoved()
        })
        // ai: shown or hidden with its parents (Compose's SwingPanel sizes it before it shows it)
        addHierarchyListener { e -> if (e.changeFlags and HierarchyEvent.SHOWING_CHANGED.toLong() != 0L) s.areaChanged() }
        addMouseListener(object : MouseAdapter() {
            override fun mouseClicked(e: MouseEvent) {
                if (!s.full && chooses()) {
                    if (listing()) hits.firstOrNull { it.first.contains(e.point) }?.second?.invoke()
                    else if (folderLink?.contains(e.point) == true) s.chooseFolder() else s.chooseFile()
                }
            }
        })
        // ai: over the list, the hand only on what takes a click; full screen's rail near the left edge
        addMouseMotionListener(object : MouseAdapter() {
            override fun mouseMoved(e: MouseEvent) {
                if (s.full) { if (e.x < EDGE_NEAR && !s.edgeRail) s.edgeRail = true; return }
                if (!listing() || !chooses()) return
                val c = if (hits.any { it.first.contains(e.point) }) Cursor.HAND_CURSOR else Cursor.DEFAULT_CURSOR
                if (cursor.type != c) cursor = Cursor.getPredefinedCursor(c)
            }
        })
        addMouseWheelListener { e -> if (listing()) { scroll += e.wheelRotation * 48; repaint() } }
        // ai: Files and folders dropped are added (a run stops first, as the web's does; refused while sending until
        // ai: 2026-10-23); the window's column takes them too (Screen.kt), and the area's outline says so while a drag is over
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

    private fun chooses() = !s.sending && !s.test
    private fun listing() = !s.test && s.items.isNotEmpty()
    private var folderLink: Rectangle? = null   // ai: where "or a folder" was painted, while it shows
    private var hits: List<Pair<Rectangle, () -> Unit>> = emptyList()   // ai: the list's controls as last painted
    private var scroll = 0

    // ai: the look for the state: a hand over the file's area, the default cursor otherwise (shown in full screen too,
    // ai: 2026-10-07); over the list, the hand on its controls alone (mouseMoved)
    fun idleLook() {
        cursor = if (chooses() && !listing()) Cursor.getPredefinedCursor(Cursor.HAND_CURSOR) else Cursor.getDefaultCursor()
        repaint()
    }

    override fun update(g: Graphics) = paint(g)
    override fun paint(g: Graphics) {
        if (presenting) return
        val g2 = g as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
        if (listing()) paintList(g2) else paintHint(g2)
        // ai: a drag over the window: the web's dashed outline, 12 px in
        if (s.dragOver) {
            g2.color = FG; g2.stroke = BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, floatArrayOf(9f, 6f), 0f)
            g2.drawRect(12, 12, width - 25, height - 25)
        }
    }

    private fun paintHint(g2: Graphics2D) {
        g2.color = SOFT   // ai: the soft fill (Parts.kt Soft)
        g2.fillRect(0, 0, width, height)
        g2.color = MUTED
        // ai: the file icon (Parts.kt Icons.file, Feather's on a 24 grid, 2 px strokes), then the line 8 px under it
        val cx = width / 2.0; val cy = height / 2.0 - 16
        drawIcon(g2, FILE_ICON, cx - 12, cy - 12)
        val text = if (s.test) "Test stream" else if (s.reading) "Reading the files" else "Choose files"
        g2.font = Font(Font.SANS_SERIF, Font.PLAIN, 14)
        val fm = g2.fontMetrics
        g2.drawString(text, (cx - fm.stringWidth(text) / 2.0).toFloat(), (cy + 12 + 8 + fm.ascent).toFloat())
        // ai: a folder instead (the web's hint link), underlined in the text's colour under the line
        folderLink = null
        if (chooses() && !s.reading) {
            val link = "or a folder"; val w = fm.stringWidth(link)
            val x = (cx - w / 2.0).toInt(); val base = (cy + 12 + 8 + fm.ascent + fm.height + 4).toInt()
            g2.color = FG
            g2.drawString(link, x, base)
            g2.stroke = BasicStroke(1f)
            g2.drawLine(x, base + 2, x + w, base + 2)
            folderLink = Rectangle(x - 8, base - fm.ascent - 4, w + 16, fm.height + 8)
        }
    }

    // ai: The selection (2026-10-23, the web's #pick): on white, a column at most 416 px wide, centred, a row an item (a
    // ai: soft square with its icon, its name, "N files · size" for a folder or its size, Remove), then Add files, Add a
    // ai: folder and Clear as the soft pills; scrolled by the wheel where it is taller than the area.
    private fun paintList(g2: Graphics2D) {
        g2.color = java.awt.Color.WHITE
        g2.fillRect(0, 0, width, height)
        val items = s.items.toList()
        val w = min(width - 32, 416); val x0 = (width - w) / 2
        val rowH = 64; val btnH = 40
        val total = items.size * rowH + 12 + btnH
        val avail = height - 32
        scroll = scroll.coerceIn(0, maxOf(0, total - avail))
        var y = if (total <= avail) (height - total) / 2 else 16 - scroll
        val got = ArrayList<Pair<Rectangle, () -> Unit>>()
        val name = Font(Font.SANS_SERIF, Font.BOLD, 15); val sub = Font(Font.SANS_SERIF, Font.PLAIN, 13)
        for ((i, it) in items.withIndex()) {
            g2.color = SOFT; g2.fillRoundRect(x0, y + 12, 40, 40, 20, 20)
            g2.color = FG; drawIcon(g2, if (it.dir) FOLDER_ICON else FILE_ICON, x0 + 8.0, y + 20.0)
            val textW = w - 52 - 48
            g2.font = name; g2.color = FG
            g2.drawString(fit(g2, it.name, textW), x0 + 52, y + 12 + 17)
            g2.font = sub; g2.color = MUTED
            val line = if (it.dir) "${it.files} ${if (it.files == 1) "file" else "files"} · ${Fmt.bytes(it.size)}${if (it.links > 0) ", ${it.links} ${if (it.links == 1) "link" else "links"} skipped" else ""}"
                else Fmt.bytes(it.size)
            g2.drawString(fit(g2, line, textW), x0 + 52, y + 12 + 36)
            // ai: Remove: an x in a 40 px square at the row's end
            val rx = x0 + w - 40; val ry = y + 12
            g2.color = FG; g2.stroke = BasicStroke(2f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
            g2.drawLine(rx + 14, ry + 14, rx + 26, ry + 26); g2.drawLine(rx + 26, ry + 14, rx + 14, ry + 26)
            val key = it.key
            got += Rectangle(rx, ry, 40, 40) to { s.remove(key) }
            if (i < items.size - 1) { g2.color = LINE; g2.fillRect(x0, y + rowH - 1, w, 1) }
            y += rowH
        }
        y += 12
        g2.font = Font(Font.SANS_SERIF, Font.BOLD, 14)
        val fm = g2.fontMetrics
        val buttons = listOf<Pair<String, () -> Unit>>("Add files" to { s.chooseFile() }, "Add a folder" to { s.chooseFolder() }, "Clear" to { s.clearPicks() })
        // ai: centred under the list (2026-10-23; from its left edge before, which read off centre)
        var x = x0 + (w - (buttons.sumOf { fm.stringWidth(it.first) + 40 } + 8 * (buttons.size - 1))) / 2
        for ((label, act) in buttons) {
            val bw = fm.stringWidth(label) + 40
            g2.color = SOFT; g2.fillRoundRect(x, y, bw, btnH, btnH, btnH)
            g2.color = FG; g2.drawString(label, x + 20, y + (btnH - fm.height) / 2 + fm.ascent)
            got += Rectangle(x, y, bw, btnH) to act
            x += bw + 8
        }
        hits = got
    }

    // ai: a line cut on a character with an ellipsis where it is wider than the room
    private fun fit(g2: Graphics2D, text: String, room: Int): String {
        val fm = g2.fontMetrics
        if (fm.stringWidth(text) <= room) return text
        var t = text
        while (t.isNotEmpty() && fm.stringWidth("$t…") > room) t = t.dropLast(1)
        return "$t…"
    }
    private fun drawIcon(g2: Graphics2D, icon: Path2D, x: Double, y: Double) {
        g2.stroke = BasicStroke(2f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
        val t = g2.transform
        g2.translate(x, y); g2.draw(icon); g2.transform = t
    }

    companion object {
        private val FG = java.awt.Color(0x11, 0x11, 0x11)
        private val MUTED = java.awt.Color(0x66, 0x66, 0x66)
        private val SOFT = java.awt.Color(0xF1, 0xF1, 0xF1)
        private val LINE = java.awt.Color(0xE4, 0xE4, 0xE4)
        // ai: Feather's file and folder (Parts.kt Icons.file; the app's ic_folder) on a 24 grid
        private val FILE_ICON = Path2D.Double().apply {
            moveTo(14.0, 2.0); lineTo(6.0, 2.0); curveTo(4.9, 2.0, 4.0, 2.9, 4.0, 4.0); lineTo(4.0, 20.0); curveTo(4.0, 21.1, 4.9, 22.0, 6.0, 22.0)
            lineTo(18.0, 22.0); curveTo(19.1, 22.0, 20.0, 21.1, 20.0, 20.0); lineTo(20.0, 8.0); closePath()
            moveTo(14.0, 2.0); lineTo(14.0, 8.0); lineTo(20.0, 8.0)
        }
        private val FOLDER_ICON = Path2D.Double().apply {
            moveTo(22.0, 19.0); curveTo(22.0, 20.1, 21.1, 21.0, 20.0, 21.0); lineTo(4.0, 21.0); curveTo(2.9, 21.0, 2.0, 20.1, 2.0, 19.0)
            lineTo(2.0, 5.0); curveTo(2.0, 3.9, 2.9, 3.0, 4.0, 3.0); lineTo(9.0, 3.0); lineTo(11.0, 6.0); lineTo(20.0, 6.0)
            curveTo(21.1, 6.0, 22.0, 6.9, 22.0, 8.0); closePath()
        }
    }
    override fun addNotify() { super.addNotify(); EventQueue.invokeLater { s.areaChanged() } }
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

    // ai: the canvas's size in device px (its user-space size by the screen's scale)
    fun devicePx(): Pair<Int, Int> {
        val t = graphicsConfiguration?.defaultTransform
        return (width * (t?.scaleX ?: 1.0)).roundToInt() to (height * (t?.scaleY ?: 1.0)).roundToInt()
    }

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
    // ai: a drag over the window (the canvas's drop target, Screen.kt's for the column): the code area's outline
    var dragOver = false
        private set
    fun dragging(on: Boolean) { if (on != dragOver) { dragOver = on; canvas.repaint() } }
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
    // ai: The transfer made at the pick (2026-10-19): the bundle's tar written and every chunk compressed (txCreate) on a
    // ai: thread of its own as soon as the choice is read, so the idle line says what goes, "<compressed> (<raw> raw)",
    // ai: and Start takes it at once (or waits for one still being made). compressed: its bytes as they go, 0 until it is
    // ai: made, kept through a run. A stop makes the next one; a new pick or the window's close lets it go.
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
    var blocks by mutableStateOf(Prefs.blocks.coerceIn(0, 128))
    var fps by mutableStateOf(if (Prefs.vsync) Pick.fpsOf(Prefs.fps) else Prefs.fps.coerceIn(1, 60))
    // ai: VSync (2026-10-23, on by default): while the code is shown its monitor holds a refresh the rate divides, the
    // ai: nearest 60 Hz, where its own does not (a 165 Hz laptop at 60: pictures of 3, 3, 2 and 3 refreshes), and is put
    // ai: back at the stop (presenter.h setVsync); off, the monitor is left as it is. FPS is Pick.FPS's chips under VSync,
    // ai: a 1 to 60 slider without it; VSync turned on takes a slider's rate to the nearest chip.
    var vsync by mutableStateOf(Prefs.vsync)
    var size by mutableStateOf(Prefs.size.coerceIn(25, 100))
    var spacing by mutableStateOf(Prefs.spacing.coerceIn(Pick.SPACING_MIN, Pick.SPACING_MAX))
    var margin by mutableStateOf(Prefs.margin.coerceIn(Pick.QUIET, Pick.MARGIN_MAX))
    var ring by mutableStateOf(Prefs.ring.let { if (it in 0..3) it else -1 })
    var codes by mutableStateOf(if (Prefs.codes == 2) 2 else 1)
    var settingsOpen by mutableStateOf(Prefs.settingsOpen)
    var toolsOpen by mutableStateOf(Prefs.toolsOpen)
    var collapsed by mutableStateOf(Prefs.collapsed)

    var phase by mutableStateOf<Phase>(Phase.Idle)
    var paused by mutableStateOf(false)
    // ai: Full screen (the Fullscreen button, F or F11; Escape, F, F11 or the rail's collapser leaves), as other apps have it
    // ai: (2026-10-23): the window manager's own (Main.kt fullscreen), the code alone in it as the web's and the app's;
    // ai: the manager's word followed (fullFromManager). Until then the column folded to the rail, which a toggle opened
    // ai: there, and the presenter bypassed the compositor.
    var full by mutableStateOf(false)
    var fullscreen: ((Boolean) -> Unit)? = null
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

    val presenting get() = p != 0L
    val sending get() = phase == Phase.On || phase == Phase.Preparing
    val canStart get() = loadError.isEmpty() && !sending && (test || (choice != null && !reading))
    val railShown get() = collapsed
    val ringIndex get() = if (ring in 0..3) ring else Pick.RING_DEFAULT
    // ai: a symbol's room in the area, device px, the Size slider's share of it (send.mjs room)
    val room get() = Pick.room(areaW, areaH, codes, ringIndex, spacing, margin) * size / 100.0
    // ai: the margin past the quiet zone the presenter keeps at the area's edges, for the ring configured
    private val edge get() = Pick.edgeShare(fmt?.span ?: Pick.span(ringIndex), margin).toFloat()
    // ai: what auto takes in the area now (0 while it has no size): the configured format's while sending
    val autoBlocks get() = fmt?.takeIf { blocks == 0 }?.let { it.subch / 8 } ?: if (areaW > 0 && areaH > 0) Pick.pick(room, ringIndex) / 8 else 0
    // ai: what the code offers at most, KB/s: codes x blocks a frame x 469 B x the pictures a second it can show, the
    // ai: asked rate or the measured refresh where lower, so a limit outside the sender shows in the figure
    // ai: (2026-10-04); null before a configure
    val capacityKBs: Double? get() {
        val f = fmt ?: return null
        val hz = present?.num("hz") ?: 0.0
        return f.codes * sentBlocks * 469.0 * (if (hz > 0) min(f.fps.toDouble(), hz) else f.fps.toDouble()) / 1000
    }
    // ai: the blocks a code of the painted format carries, from the sender's stats, the configured format's before the
    // ai: first stats; its profile, "" none
    val sentBlocks: Int get() = sender?.num("blocks")?.toInt()?.takeIf { it > 0 } ?: fmt?.let { Native.blocksFor(it.subch) } ?: 0
    val tiers: String get() = sender?.str("tiers").orEmpty()

    // ai: The chooser (2026-10-23; Swing's folder chooser before, which looks like no desktop's): the
    // ai: desktop's own dialog through the XDG desktop portal (Native.portalPick: GTK's on Cinnamon and GNOME, KDE's on
    // ai: Plasma, attached to this window, several files or several folders at once), on a thread of its own while it is
    // ai: open; where no portal answers, zenity's (GTK's too); else, as before, AWT's FileDialog for files (GTK's) and
    // ai: Swing's JFileChooser for folders, which AWT's cannot pick. One open at a time.
    private var choosing = false
    fun chooseFile() = choose(false)
    fun chooseFolder() = choose(true)
    private fun choose(folders: Boolean) {
        if (choosing) return
        choosing = true
        val parent = runCatching { Native.windowOf(canvas) }.getOrDefault(0L)
        val title = if (folders) "Choose folders" else "Choose files"
        thread(name = "lizard-chooser", isDaemon = true) {
            val got = portal(parent, title, folders) ?: zenity(title, folders)
            EventQueue.invokeLater {
                choosing = false
                if (got == null) fallback(folders) else if (got.isNotEmpty()) pick(got)
            }
        }
    }
    // ai: the portal's choice as files (its file:// URIs; Windows' dialog gives paths), none when cancelled, null where it
    // ai: could not be asked
    private fun portal(parent: Long, title: String, folders: Boolean): List<File>? {
        if (Native.mac) return null
        val uris = runCatching { Native.portalPick(parent, title, folders, true) }.getOrNull()
        if (uris == null) { System.err.println("chooser: no portal: ${runCatching { Native.txError() }.getOrDefault("")}"); return null }
        return uris.mapNotNull { u -> runCatching { if (u.startsWith("file:")) File(java.net.URI(u)) else File(u) }.getOrNull() }
    }
    // ai: zenity's, a path a line; none when cancelled (exit 1), null where it is not there
    private fun zenity(title: String, folders: Boolean): List<File>? = if (Native.mac || Native.windows) null else runCatching {
        val p = ProcessBuilder(listOf("zenity", "--file-selection", "--multiple", "--separator=\n", "--title=$title") + (if (folders) listOf("--directory") else emptyList()))
            .redirectError(ProcessBuilder.Redirect.DISCARD).start()
        val out = p.inputStream.bufferedReader().readText()
        when (p.waitFor()) { 0 -> out.lines().filter { it.isNotEmpty() }.map(::File); 1 -> emptyList(); else -> null }
    }.getOrNull()
    // ai: the dialogs before the portal: AWT's FileDialog (GTK's on Linux, the system's open panel on macOS), one file or
    // ai: several, over the sender's window (the canvas's frame); a folder through Swing's chooser, which AWT's dialog
    // ai: cannot pick, but on macOS, where the panel takes folders when asked
    private fun fallback(folders: Boolean) {
        if (Native.mac) {
            System.setProperty("apple.awt.fileDialogForDirectories", folders.toString())
            val d = FileDialog(SwingUtilities.getWindowAncestor(canvas) as? Frame, if (folders) "Choose folders" else "Choose files", FileDialog.LOAD)
            d.isMultipleMode = true
            d.isVisible = true
            System.setProperty("apple.awt.fileDialogForDirectories", "false")
            if (d.files.isNotEmpty()) pick(d.files.toList())
        } else if (!folders) {
            val d = FileDialog(SwingUtilities.getWindowAncestor(canvas) as? Frame, "Choose files", FileDialog.LOAD)
            d.isMultipleMode = true
            d.isVisible = true
            if (d.files.isNotEmpty()) pick(d.files.toList())
        } else {
            val c = JFileChooser().apply { fileSelectionMode = JFileChooser.DIRECTORIES_ONLY; dialogTitle = "Choose a folder" }
            if (c.showOpenDialog(SwingUtilities.getWindowAncestor(canvas)) == JFileChooser.APPROVE_OPTION) c.selectedFile?.let { pick(listOf(it)) }
        }
    }

    // ai: Files and folders added (picked, dropped, handed in), walked off the UI thread
    fun pick(files: List<File>) {
        if (sending) stop()
        if (phase is Phase.Error) phase = Phase.Idle
        walks++; readingNow(); canvas.idleLook()
        thread(name = "lizard-send-pick", isDaemon = true) {
            val got = runCatching { files.map { itemOf(it) } }
            EventQueue.invokeLater {
                walks--; readingNow()
                got.onSuccess { list ->
                    val have = items.mapTo(HashSet()) { it.key }
                    val fresh = list.filter { have.add(it.key) }
                    if (fresh.isNotEmpty()) { items.addAll(fresh); changed() }
                }.onFailure { phase = Phase.Error("The files could not be read: ${it.message}") }
                canvas.idleLook()
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
    // ai: the selection changed: the run stopped, the made transfer let go, the bundle made again off the UI thread and
    // ai: its transfer made (make)
    private fun changed() {
        if (sending) stop()
        dropMade()
        if (phase is Phase.Error) phase = Phase.Idle
        choice = null; singleFile = null; fileSize = 0; compressed = 0
        val list = items.toList()
        val my = ++bundling
        bundleNow = list.isNotEmpty(); readingNow(); canvas.idleLook()
        if (list.isEmpty()) return
        thread(name = "lizard-send-bundle", isDaemon = true) {
            val b = Bundle(list.flatMap { it.entries })
            EventQueue.invokeLater {
                if (my != bundling) return@invokeLater
                bundleNow = false; readingNow()
                choice = b; singleFile = if (b.single != null) list[0].file else null; fileSize = b.size
                make()
                canvas.idleLook()
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

    // ai: Start as the person presses it (the column's Start, the rail's play): before the first send, the brightness tip
    // ai: (2026-10-04; the web sender's the same): the monitor's contrast and brightness are the channel's and no app
    // ai: sets them (2026-10-03, the S26 by hand: the contrast at its maximum took 2:1, two LIZARD-416 under one code
    // ai: rate, past 2.90 MB/s). Shown until a send goes from it (its Start),
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
        runTest = test
        runName = if (test) "the test stream" else choice!!.name
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
        dropTar()
        startWhenMade = false
        if (sending) phase = Phase.Idle
        paused = false; fmt = null; present = null; sender = null
        // ai: the next Start's transfer, made afresh (the stopped one went with its run)
        if (!closing && choice != null && made == 0L && !makingNow) EventQueue.invokeLater { if (!closing && !sending && choice != null && made == 0L && !makingNow) make() }
    }
    // ai: the window closing: the run stopped and the made transfer let go
    fun close() { closing = true; stop(); dropMade() }

    private fun fail(why: String) { stop(); phase = Phase.Error(why) }

    // ai: play and pause as a video player's (the rail's button, Space and K): Start where nothing runs (the tip first
    // ai: where it is due), else Pause or Resume
    fun playPause() { if (phase == Phase.On) pause(!paused) else if (!sending) requestStart() }
    // ai: Paused: the last picture stays (the presenter takes no new one); a receiver holds what it has.
    fun pause(on: Boolean) {
        if (phase != Phase.On) return
        paused = on
        if (p != 0L) Native.presentPause(p, on)
    }

    fun toggleFull() { full = !full; edgeRail = false; fullscreen?.invoke(full) }
    // ai: the manager's own word on the window (its shortcut, a leave of its own, a request of ours taken): followed,
    // ai: nothing asked
    fun fullFromManager(on: Boolean) { if (on != full) { full = on; edgeRail = false } }

    fun toggleRail() { collapsed = !collapsed; Prefs.collapsed = collapsed }
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
    fun chooseVsync(on: Boolean) {
        vsync = on; Prefs.vsync = on
        if (on && fps !in Pick.FPS) { chooseFps(Pick.fpsOf(fps)); commitFps() }
        if (p != 0L) Native.presentVsync(p, on)
    }
    fun chooseSize(v: Int) { size = v.coerceIn(25, 100); Prefs.size = size }
    fun chooseSpacing(v: Int) { spacing = v.coerceIn(Pick.SPACING_MIN, Pick.SPACING_MAX); Prefs.spacing = spacing }
    fun chooseMargin(v: Int) { margin = v.coerceIn(Pick.QUIET, Pick.MARGIN_MAX); Prefs.margin = margin }
    fun chooseRing(v: Int) { ring = if (v in 0..3) v else -1; Prefs.ring = ring; repick() }
    fun chooseCodes(v: Int) { codes = if (v == 2) 2 else 1; Prefs.codes = codes; repick() }
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

    // ai: macOS: the presenter's layer follows the canvas (a move or a resize of it or of the window)
    fun canvasMoved() { if (p != 0L) Native.presentPlace(p, canvas.place()) }

    // ai: the canvas's window going: the presenter stopped first (its surface is that window); begun again on the next
    fun canvasGone() {
        if (p != 0L) { Native.presentStop(p); p = 0L }
        canvas.presenting = false
    }

    private fun begin() {
        if (h == 0L || p != 0L || !canvas.isShowing || areaW <= 0 || areaH <= 0) return
        if (!configure()) return
        canvas.presenting = true
        val pr = Native.presentStart(canvas, h, fps, size / 100f, edge, false, vsync, canvas.place())
        if (pr == 0L) { fail("The code cannot be shown: ${Native.txError()}"); return }
        p = pr
        if (paused) Native.presentPause(p, true)
        timer.start()
    }

    // ai: The format the area holds now, configured where it changed: blocks set by hand win, else the largest the room
    // ai: holds (Pick, the web's pickVersion). false where it was refused (the run then ends with why).
    private fun configure(): Boolean {
        if (h == 0L) return false
        val subch = if (blocks > 0) 8 * blocks else Pick.pick(room, ringIndex)
        val f = Format(Pick.nFor(subch), subch, Pick.span(ringIndex), fps, codes, if (codes > 1) Pick.gapOf(spacing) else 0,
            when (enc) { "cpu" -> 0; "gpu" -> 1; else -> 2 })
        if (f == fmt) return true
        val threads = min(8, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val err = Native.txConfigure(h, f.n, f.subch, f.span, f.fps, f.codes, f.gap, threads, f.painter, Native.assets)
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
        if (p != 0L) { presentRaw = Native.presentStats(p); present = parse(presentRaw) }
        val e = sender?.str("error").orEmpty().ifEmpty { if (p != 0L) present?.str("error").orEmpty() else "" }
        if (e.isNotEmpty()) fail(e)
    }

    // ai: what goes, "<compressed> (<raw> raw)" once the transfer is made, "<raw> raw" while it is being made (the web's
    // ai: and the app's the same)
    // ai: "raw" kept with its figure (a no-break space, 2026-10-23: "(47 KB / raw)" broke there)
    private fun sizeText() = if (compressed > 0) "${Fmt.bytes(compressed)} (${Fmt.bytes(fileSize)}${Fmt.NB}raw)" else "${Fmt.bytes(fileSize)}${Fmt.NB}raw"

    // ai: The state line (the web's #state): what is sent, paused, or the error (red; Main.kt colours it). A line naming
    // ai: the file says so in three parts, the name between (2026-10-23), for Parts.kt NameLine to hold its head on one
    // ai: line; any other is all `before`.
    data class StateLine(val before: String, val name: String? = null, val after: String = "")
    val stateParts: StateLine get() = when (val ph = phase) {
        is Phase.Error -> StateLine(ph.why)
        Phase.Preparing -> if (runTest) StateLine("Preparing $runName") else StateLine("Preparing ", runName)
        Phase.On -> if (paused) StateLine("Paused") else if (runTest) StateLine("Sending $runName") else StateLine("Sending ", runName, ", ${sizeText()}")
        // ai: idle, the web's "<name>, <size>" (sizeText) or the app's "The test stream"
        Phase.Idle -> if (loadError.isNotEmpty()) StateLine(loadError) else if (test) StateLine("The test stream") else if (reading) StateLine("Reading the files")
            else choice?.let { StateLine("", it.name, ", ${if (it.single == null) "${it.files} ${if (it.files == 1) "file" else "files"}, " else ""}${sizeText()}") } ?: StateLine("")
    }
    val stateLine: String get() = stateParts.let { "${it.before}${it.name ?: ""}${it.after}" }

    // ai: The figures (the web's #nums): the rate the pictures actually presented carry (the presenter's fresh pictures a
    // ai: second x codes x blocks a frame x 469 B, so a limit outside the sender shows in the figure; 2026-10-04); the
    // ai: pass, beside it until 2026-10-05, is the readout's.
    val figures: String get() {
        if (phase != Phase.On || paused) return ""
        val f = fmt ?: return ""
        val pr = present ?: return ""
        return Fmt.rate(pr.num("shownFps") * f.codes * sentBlocks * 469 / 1000.0)
    }

    // ai: The readout (the web's #tx, Developer Tools): the format and why, the file, the frame on the surface, a frame's
    // ai: bytes, the pictures shown against the asked rate and the screen's, the refreshes each held, the encoder.
    val readout: String get() {
        val f = fmt ?: return ""
        val sd = sender ?: return ""
        val pr = present
        val b = sentBlocks
        val l = mutableListOf<String>()
        l += "LIZARD-${f.subch} ($b blocks${if (tiers.isNotEmpty()) ": $tiers" else ""}), picture ${f.n} samples in the ${f.span / 2}-cell ring" +
            if (blocks > 0) ", set by hand" else ", chosen from a ${room.roundToInt()} px room and nothing else"
        if (!runTest) l += "file $fileSize B${sd.num("sentBytes").toLong().let { if (it in 1 until fileSize) ", $it B as sent (zstd)" else "" }}: ${sd.num("chunks").toInt()} chunk${if (sd.num("chunks").toInt() == 1) "" else "s"}, " +
            "${sd.num("lap").toInt()} data blocks a pass, BLAKE3 ${sd.str("root").take(16)}..., header in the light" +
            (sd.num("pass").let { p -> if (p >= 1) "\npass ${floor(p).toInt()}" + (sd.num("offeredKBs").let { k -> if (k > 0 && sd.num("lap") > 0) ", ${Fmt.f1(sd.num("lap") / (k * 1000 / 469))} s a pass" else "" }) else "" })
        l += "frame ${sd.num("width").toInt()} x ${sd.num("side").toInt()} px" + (if (f.codes > 1) ", ${f.codes} side by side $spacing modules apart" else "") + ", margin $margin modules" +
            (pr?.let { " on a ${it.str("surface").replace("x", " x ")} px surface" } ?: "")
        l += "${if (f.codes > 1) "${f.codes} x " else ""}$b x 469 B = ${f.codes * b * 469} B per frame"
        if (pr != null) {
            l += "shown ${Fmt.f1(pr.num("shownFps"))}/s of ${f.fps} asked, ${Fmt.f1(pr.num("presentsPerSec"))} presents a second, the screen " +
                "${Fmt.f1(pr.num("hz"))} Hz${if (pr.num("modeHz") > 0) " (its mode ${Fmt.f1(pr.num("modeHz"))})" else ""}" +
                (if (pr.num("modeFrom") > 0) ", held by VSync from ${Fmt.f1(pr.num("modeFrom"))} Hz" else "") + ", times from ${if (pr.flag("presentWait")) "present_wait" else "the acquires"}"
            val held = (pr["held"] as? JsonObject)?.entries?.sortedBy { it.key.toIntOrNull() ?: Int.MAX_VALUE }
                ?.joinToString(", ") { "${it.key}: ${(it.value as? JsonPrimitive)?.contentOrNull ?: "?"}" }.orEmpty()
            l += "pictures by the refreshes each held: ${held.ifEmpty { "none" }}; missed ${pr.num("missed").toInt()}, behind the painter ${pr.num("behind").toInt()}"
            l += "presenter: Vulkan on ${pr.str("device")}${if (pr.flag("paused")) ", paused" else ""}" +
                when ((pr["direct"] as? JsonPrimitive)?.booleanOrNull) { true -> ", direct"; false -> ", composited"; null -> "" }
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
