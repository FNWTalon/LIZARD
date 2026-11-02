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
import java.awt.BasicStroke
import java.awt.Canvas
import java.awt.Font
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
import java.awt.Rectangle
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
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import dev.lizard.ui.*
import kotlinx.coroutines.runBlocking

// ai: The code area (the web's <main>): a heavyweight AWT canvas, a native window of its own inside the Compose window
// ai: (Screen.kt's SwingPanel), on which the presenter's swapchain draws; white, and nothing painted over the
// ai: presenter's pictures by AWT. Its size in device px is the room the format is picked from; it tells the state when
// ai: it has a window (addNotify), is shown or resized, and, before that window goes (removeNotify: a dispose, a window
// ai: made again), stops the presenter on it. A click on it does nothing in full screen (2026-10-24: Escape, F, F11 and
// ai: the rail's collapser leave it); the Idle
// ai: (2026-10-07, the web's middle of its page and the app's code box) it is the file's area: the file upload (since
// ai: 2026-10-26 the File and Folder tiles and the selection's card, paintUpload), or "Test stream"; files and folders
// ai: dropped on it are taken (the test stream off; several, or a folder, one tar since 2026-10-16). AWT paints that
// ai: itself, since nothing of Compose can be drawn over a heavyweight canvas.
class CodeCanvas(private val s: SendState) : Canvas() {
    @Volatile var presenting = false
    // ai: how near the left edge the pointer shows full screen's rail, the window's units
    private val EDGE_NEAR = 24

    init {
        background = java.awt.Color.WHITE
        isFocusable = false
        addComponentListener(object : ComponentAdapter() {
            // ai: a resize paints the idle area too, at once: nothing else does on a heavyweight canvas that shrinks (no
            // ai: expose), so the file upload stayed at the old size until the next change (2026-10-26); a queued repaint
            // ai: left the old size's pixels smeared over the new area while the column's edge was dragged (2026-11-02)
            override fun componentResized(e: ComponentEvent) { s.areaChanged(); s.canvasMoved(); if (!presenting) paintNow() }
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
                if (!s.full && !nearEdge(e) && chooses()) hits.firstOrNull { it.first.contains(e.point) }?.second?.invoke()
            }
            override fun mousePressed(e: MouseEvent) {
                if (e.button == MouseEvent.BUTTON1 && nearEdge(e)) { pressX = e.xOnScreen; pressed = true }
            }
            override fun mouseReleased(e: MouseEvent) {
                pressed = false
                if (gripped) { gripped = false; grip?.done(); grip?.hover(nearEdge(e)) }
            }
            override fun mouseExited(e: MouseEvent) { if (!gripped) grip?.hover(false) }
        })
        // ai: over the file upload, the hand only on what takes a click; the left-right resize cursor over the column's
        // ai: edge; full screen's rail near the left edge
        addMouseMotionListener(object : MouseAdapter() {
            override fun mouseMoved(e: MouseEvent) {
                // ai: full screen: nothing here takes a click, so the arrow (a resize cursor left from the column's edge stuck
                // ai: there, 2026-11-02), and the rail near the left edge
                if (s.full && !s.leased) {
                    if (cursor.type != Cursor.DEFAULT_CURSOR) cursor = Cursor.getDefaultCursor()
                    if (e.x < EDGE_NEAR && !s.edgeRail) s.edgeRail = true
                    return
                }
                val near = nearEdge(e)
                grip?.hover(near)
                val c = when {
                    near -> Cursor.E_RESIZE_CURSOR
                    uploading() && chooses() && hits.any { it.first.contains(e.point) } -> Cursor.HAND_CURSOR
                    else -> Cursor.DEFAULT_CURSOR
                }
                if (cursor.type != c) cursor = Cursor.getPredefinedCursor(c)
            }
            // ai: the edge dragged by the pointer's move on the screen (the canvas moves under it as the column resizes);
            // ai: AWT's units are the window's logical pixels, Compose's dp on the desktop. Taken at the first move, as
            // ai: Compose's drag is, so a press and release alone neither saves the width nor re-picks.
            override fun mouseDragged(e: MouseEvent) {
                val g = grip ?: return
                if (!pressed) return
                val dx = e.xOnScreen - pressX
                if (dx == 0) return
                if (!gripped) { gripped = true; g.start() }
                pressX = e.xOnScreen
                g.drag(dx.dp)
            }
        })
        addMouseWheelListener { e -> if (uploading()) { scroll += e.wheelRotation * 48; repaint() } }
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
    // ai: the column's edge over the canvas's left EdgeStrip less EdgeRoom (dev.lizard.ui.EdgeGrip: Compose neither draws
    // ai: nor hears over this heavyweight), given by the code area while the column is open, null otherwise
    // ai: (DesktopSend.CodeArea)
    var grip: EdgeGrip? = null
    private var pressed = false   // ai: the first button went down on the edge
    private var gripped = false   // ai: and the edge is being dragged
    private var pressX = 0
    private fun nearEdge(e: MouseEvent) = grip != null && !s.full && e.x < (EdgeStrip - EdgeRoom).value
    // ai: a pick being walked or the bundle made again (`reading`, which every add, remove and Clear sets) keeps the upload
    // ai: up, the column's state line saying "Reading the files" (a cover took the area for it until 2026-10-26: a flash)
    private fun uploading() = !s.test
    private var hits: List<Pair<Rectangle, () -> Unit>> = emptyList()   // ai: the file upload's controls as last painted
    private var scroll = 0

    // ai: the look for the state, painted: the hand (mouseMoved's, over the file upload's controls) given back to the
    // ai: arrow only where nothing here takes a click any more, full screen included (2026-11-02: every change of the
    // ai: state reset it under a pointer resting on a tile)
    fun idleLook() {
        if (s.full || !uploading() || !chooses()) cursor = Cursor.getDefaultCursor()
        repaint()
    }
    // ai: Painted now, not queued (2026-11-02): where the presenter has just let go of the window (Stop) or the canvas has
    // ai: just been resized, a queued repaint left the last picture, or the old size's pixels, up until AWT came to it
    fun paintNow() {
        if (!isShowing) return
        val g = graphics ?: return
        try { paint(g) } finally { g.dispose() }
    }

    override fun update(g: Graphics) = paint(g)
    // ai: Drawn off screen at the screen's pixels, then put on screen in one copy (2026-10-26): drawn straight onto the
    // ai: heavyweight canvas, each repaint (a wheel step, a removal) showed the area cleared and half drawn for a moment.
    // ai: One buffer, made again only to grow (a dragged edge resized it at each step), its top left corner used.
    private var back: BufferedImage? = null
    override fun paint(g: Graphics) {
        // ai: the code shown on a monitor of its own: the area here plain, a line saying where (the window is the control)
        if (s.leased) { paintLeased(g as Graphics2D); return }
        if (presenting || width <= 0 || height <= 0) return
        val t = graphicsConfiguration?.defaultTransform
        val sx = t?.scaleX ?: 1.0; val sy = t?.scaleY ?: 1.0
        val bw = (width * sx).roundToInt().coerceAtLeast(1); val bh = (height * sy).roundToInt().coerceAtLeast(1)
        val img = back?.takeIf { it.width >= bw && it.height >= bh } ?: BufferedImage(bw, bh, BufferedImage.TYPE_INT_RGB).also { back = it }
        val g2 = img.createGraphics()
        try { g2.clipRect(0, 0, bw, bh); g2.scale(sx, sy); draw(g2) } finally { g2.dispose() }
        g.drawImage(img, 0, 0, width, height, 0, 0, bw, bh, null)
    }
    private fun draw(g2: Graphics2D) {
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
        if (uploading()) paintUpload(g2) else paintHint(g2)
        // ai: a drag over the window: the web's dashed outline, 12 px in
        if (s.dragOver) {
            g2.color = FG; g2.stroke = BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, floatArrayOf(9f, 6f), 0f)
            g2.drawRect(12, 12, width - 25, height - 25)
        }
    }

    // ai: the control window's area while the code plays on a monitor of its own: "Playing on <monitor>" (or "Paused
    // ai: on"), centred on the soft fill
    private fun paintLeased(g2: Graphics2D) {
        g2.color = SOFT
        g2.fillRect(0, 0, width, height)
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
        g2.color = FG
        g2.font = face(Font.PLAIN, 16)
        val text = "${if (s.paused) "Paused" else "Playing"} on ${s.leasedOn}"
        val fm = g2.fontMetrics
        g2.drawString(text, ((width - fm.stringWidth(text)) / 2.0).toFloat(), ((height - fm.height) / 2.0 + fm.ascent).toFloat())
    }

    // ai: the test stream: the soft fill, the file icon and the line 8 px under it
    private fun paintHint(g2: Graphics2D) {
        g2.color = SOFT   // ai: the soft fill (the theme's Soft)
        g2.fillRect(0, 0, width, height)
        g2.color = MUTED
        val cx = width / 2.0; val cy = height / 2.0 - 16
        drawIcon(g2, FILE_ICON, cx - 12, cy - 12)
        val text = "Test stream"
        g2.font = face(Font.PLAIN, 14)
        val fm = g2.fontMetrics
        g2.drawString(text, (cx - fm.stringWidth(text) / 2.0).toFloat(), (cy + 12 + 8 + fm.ascent).toFloat())
    }

    // ai: The file upload (2026-10-26, LocalSend's geometry; the web's #pick): on white, a column at most 576 px wide
    // ai: centred in the canvas, the whole centred in its height where it fits, else from 16 px down (the phone's shared
    // ai: Upload stretches across its area at the top; a desktop window keeps the centred column, 2026-11-01): the File and
    // ai: Folder tiles (104 px tall on the soft fill, an icon over one word, one row in equal cells), and under them, while
    // ai: the selection holds something, its card as wide (the card fill): the count with Clear, then a tile an item on
    // ai: white (its icon, its name with the stem cut and the ending kept, "N files" or its size, an x at its corner that
    // ai: removes it), in rows stretched across the card, 104 px the narrowest before one folds; the card takes what height
    // ai: is left (a row of items at least) and only its items scroll, down, by the wheel; 12 px under the card, centred,
    // ai: what goes (choiceLine: the bundle's name, its stem cut and its ending kept, then its files and the bytes that go;
    // ai: its room kept while the pick is walked or the bundle made; 2026-11-01, the column's state line before). The
    // ai: tiles' words, the count, Clear, the items' names and the name under the card bold, the other lines plain.
    private fun paintUpload(g2: Graphics2D) {
        g2.color = java.awt.Color.WHITE
        g2.fillRect(0, 0, width, height)
        val items = s.items.toList()
        val t = 104; val gap = 12; val ig = 8
        val cardW = min(width - 32, 576)
        val perRow = maxOf(1, (cardW - 24 + ig) / (t + ig))
        val iw = (cardW - 24 - (perRow - 1) * ig) / perRow
        val rows = (items.size + perRow - 1) / perRow
        val avail = height - 32
        val gridH = rows * t + (rows - 1) * ig
        g2.font = face(Font.BOLD, 15)
        val lineH = 12 + g2.fontMetrics.height   // ai: the line under the card and its gap
        val cardH = if (items.isEmpty()) 0 else minOf(4 + 48 + 4 + gridH + 12, maxOf(avail - t - 16 - lineH, 4 + 48 + 4 + t + 12))
        val viewH = cardH - (4 + 48 + 4 + 12)   // ai: the items' window inside the card
        scroll = scroll.coerceIn(0, maxOf(0, gridH - viewH))
        val total = t + if (items.isEmpty()) 0 else 16 + cardH + lineH
        var y = if (total <= avail) (height - total) / 2 else 16
        val got = ArrayList<Pair<Rectangle, () -> Unit>>()
        val tiles = listOf<Triple<Path2D, String, () -> Unit>>(Triple(FILE_ICON, "File") { s.chooseFile() }, Triple(FOLDER_ICON, "Folder") { s.chooseFolder() })
        val tw = (cardW - (tiles.size - 1) * gap) / tiles.size
        var x = (width - cardW) / 2
        g2.font = face(Font.BOLD, 15)
        for ((icon, label, act) in tiles) {
            g2.color = SOFT; g2.fillRoundRect(x, y, tw, t, 24, 24)
            g2.color = FG; drawIcon(g2, icon, x + tw / 2.0 - 16, y + 22.0, 32.0 / 24)
            val fm = g2.fontMetrics
            g2.drawString(label, x + (tw - fm.stringWidth(label)) / 2, y + 22 + 32 + 8 + fm.ascent)
            got += Rectangle(x, y, tw, t) to act
            x += tw + gap
        }
        if (items.isNotEmpty()) {
            y += t + 16
            val cx = (width - cardW) / 2
            g2.color = CARD; g2.fillRoundRect(cx, y, cardW, cardH, 24, 24)
            g2.font = face(Font.BOLD, 14); g2.color = FG
            var fm = g2.fontMetrics
            g2.drawString("${items.size} ${if (items.size == 1) "item" else "items"}", cx + 16, y + 4 + (48 - fm.height) / 2 + fm.ascent)
            g2.font = face(Font.BOLD, 15); fm = g2.fontMetrics
            val clear = Rectangle(cx + cardW - 12 - (fm.stringWidth("Clear") + 24), y + 8, fm.stringWidth("Clear") + 24, 40)
            g2.drawString("Clear", clear.x + 12, clear.y + (40 - fm.height) / 2 + fm.ascent)
            got += clear to { s.clearPicks() }
            s.choiceLine?.let { (name, after) ->
                g2.font = face(Font.PLAIN, 15); val aw = g2.fontMetrics.stringWidth(after)
                g2.font = face(Font.BOLD, 15); fm = g2.fontMetrics
                val shown = fitName(g2, name, maxOf(cardW - aw, cardW / 2))
                val nw = fm.stringWidth(shown)
                g2.font = face(Font.PLAIN, 15)
                val tail = fit(g2, after, cardW - nw)
                val lx = cx + (cardW - nw - g2.fontMetrics.stringWidth(tail)) / 2; val ly = y + cardH + 12 + fm.ascent
                g2.font = face(Font.BOLD, 15); g2.color = FG; g2.drawString(shown, lx, ly)
                g2.font = face(Font.PLAIN, 15); g2.color = MUTED; g2.drawString(tail, lx + nw, ly)
            }
            val top = y + 4 + 48 + 4
            val view = Rectangle(cx, top, cardW, viewH)
            val clip = g2.clip
            g2.clip(view)
            for ((i, it) in items.withIndex()) {
                val ix = cx + 12 + (i % perRow) * (iw + ig); val iy = top + (i / perRow) * (t + ig) - scroll
                if (iy + t < top || iy > top + viewH) continue
                g2.color = java.awt.Color.WHITE; g2.fillRoundRect(ix, iy, iw, t, 24, 24)
                g2.color = MUTED; drawIcon(g2, if (it.dir) FOLDER_ICON else FILE_ICON, ix + iw / 2.0 - 14, iy + 18.0, 28.0 / 24)
                g2.font = face(Font.BOLD, 13); g2.color = FG; fm = g2.fontMetrics
                val name = fitName(g2, it.name, iw - 16)
                g2.drawString(name, ix + (iw - fm.stringWidth(name)) / 2, iy + 18 + 28 + 6 + fm.ascent)
                val nameBottom = iy + 18 + 28 + 6 + fm.height
                g2.font = face(Font.PLAIN, 12); g2.color = MUTED; fm = g2.fontMetrics
                val sub = fit(g2, if (it.dir) "${it.files} ${if (it.files == 1) "file" else "files"}" else Readout.bytes(it.size), iw - 16)
                g2.drawString(sub, ix + (iw - fm.stringWidth(sub)) / 2, nameBottom + fm.ascent)
                // ai: Remove: an x in a 28 px square at the tile's top right
                val rx = ix + iw - 30; val ry = iy + 2
                g2.color = FG; g2.stroke = BasicStroke(2f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
                g2.drawLine(rx + 9, ry + 9, rx + 19, ry + 19); g2.drawLine(rx + 19, ry + 9, rx + 9, ry + 19)
                val key = it.key
                Rectangle(rx, ry, 28, 28).intersection(view).takeIf { !it.isEmpty }?.let { r -> got += r to { s.remove(key) } }
            }
            g2.clip = clip
        }
        hits = got
    }

    // ai: a name cut to the room as the web's .fname and the Android app's NameLine cut it: the stem with an ellipsis, its
    // ai: ending (Readout.nameParts) kept whole
    private fun fitName(g2: Graphics2D, name: String, room: Int): String {
        val fm = g2.fontMetrics
        if (fm.stringWidth(name) <= room) return name
        val (stem, end) = Readout.nameParts(name)
        var t = stem
        while (t.isNotEmpty() && fm.stringWidth("$t…$end") > room) t = t.dropLast(1)
        return if (t.isEmpty()) fit(g2, name, room) else "$t…$end"
    }

    // ai: a line cut on a character with an ellipsis where it is wider than the room
    private fun fit(g2: Graphics2D, text: String, room: Int): String {
        val fm = g2.fontMetrics
        if (fm.stringWidth(text) <= room) return text
        var t = text
        while (t.isNotEmpty() && fm.stringWidth("$t…") > room) t = t.dropLast(1)
        return "$t…"
    }
    // ai: an icon on its 24 grid at (x, y), k times its size, its 2 px strokes scaled with it (as an svg.i drawn larger)
    private fun drawIcon(g2: Graphics2D, icon: Path2D, x: Double, y: Double, k: Double = 1.0) {
        g2.stroke = BasicStroke(2f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
        val t = g2.transform
        g2.translate(x, y); g2.scale(k, k); g2.draw(icon); g2.transform = t
    }

    companion object {
        // ai: Inter (the shared theme's face, the web's: dev.lizard.ui interBytes) for the text AWT paints here, the
        // ai: variable file at its default instance, bold derived; the platform's sans where it does not load
        // ai: (2026-10-25: that sans read beside Compose's Inter until then)
        private val inter: Font? = runCatching {
            Font.createFont(Font.TRUETYPE_FONT, java.io.ByteArrayInputStream(runBlocking { interBytes() }))
        }.getOrNull()
        private fun face(style: Int, size: Int): Font = inter?.deriveFont(style, size.toFloat()) ?: Font(Font.SANS_SERIF, style, size)
        // ai: the shared theme's colours as AWT's
        private fun awt(c: androidx.compose.ui.graphics.Color) = java.awt.Color(c.toArgb(), true)
        private val FG = awt(Fg)
        private val MUTED = awt(Muted)
        private val SOFT = awt(Soft)
        private val CARD = awt(Card)
        // ai: Feather's file and folder (the shared Icons.file and Icons.folder) on a 24 grid
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
    // ai: its window made (Send entered): the area read and the idle look painted at once, not at AWT's first repaint
    // ai: its window made (Send entered): its X background white (Native.windowBackground: AWT's, the theme's grey, showed
    // ai: until the first paint), the area read and the idle look painted at once, not at AWT's first repaint
    override fun addNotify() {
        super.addNotify()
        if (s.loadError.isEmpty()) runCatching { Native.windowBackground(this, 0xFFFFFF) }
        EventQueue.invokeLater { s.areaChanged(); if (!presenting) paintNow() }
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

    // ai: the canvas's size in device px (its user-space size by the screen's scale)
    fun devicePx(): Pair<Int, Int> {
        val t = graphicsConfiguration?.defaultTransform
        return (width * (t?.scaleX ?: 1.0)).roundToInt() to (height * (t?.scaleY ?: 1.0)).roundToInt()
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
    var leasedOn = ""
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
    private val leaseTimer = Timer(200) { if (leased && p != 0L && parse(Native.presentStats(p))?.str("direct").isNullOrEmpty()) leaseEnded() }

    val presenting get() = p != 0L
    val sending get() = phase == Phase.On || phase == Phase.Preparing
    val canStart get() = loadError.isEmpty() && !sending && (test || (choice != null && !reading))
    val railShown get() = collapsed
    val ringIndex get() = if (ring in Pick.RINGS.indices) ring else Pick.RING_DEFAULT
    // ai: a symbol's room in the area, device px, the Size slider's share of it (send.mjs room)
    val room get() = Room.room(areaW, areaH, codes, ringIndex, spacing, margin) * size / 100.0
    // ai: the margin past the quiet zone the presenter keeps at the area's edges, for the ring configured
    private val edge get() = Room.edgeShare(fmt?.span ?: Pick.span(ringIndex), margin).toFloat()
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
    fun chooseFile() = Chooser.choose(canvas, "Choose files", false, true) { pick(it) }
    fun chooseFolder() = Chooser.choose(canvas, "Choose folders", true, true) { pick(it) }

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
    // ai: the selection changed: the run stopped, the made transfer let go, the bundle made again off the UI thread (its
    // ai: transfer waits for Start)
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
                areaChanged()
            }
        }
    }

    fun stop() {
        gen++
        timer.stop()
        settle.stop()
        if (p != 0L) { Native.presentStop(p); p = 0L }
        if (leased) leaseEnded()
        canvas.presenting = false
        canvas.paintNow()
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
    // ai: still has a window, the tip put away, the run stopped (the presenter's canvas leaves the window with the
    // ai: screen); the made transfer kept for a return
    fun leave() {
        if (full) toggleFull()
        askBrightness = false; edgeRail = false
        if (sending) stop()
    }

    private fun fail(why: String) { stop(); phase = Phase.Error(why) }

    // ai: play and pause as a video player's (the rail's button, Space and K): Start where nothing runs (the tip first
    // ai: where it is due), else Pause or Resume
    fun playPause() { if (phase == Phase.On) pause(!paused) else if (!sending) requestStart() }
    // ai: Paused: the last picture stays (the presenter takes no new one); a receiver holds what it has.
    fun pause(on: Boolean) {
        if (phase != Phase.On) return
        paused = on
        if (p != 0L) Native.presentPause(p, on)
        if (leased) canvas.repaint()
        if (!on) takeOver()
    }

    fun toggleFull() {
        edgeRail = false
        if (leased) { if (p != 0L) { Native.presentStop(p); p = 0L }; leaseEnded(); if (sending) begin(); return }
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
    // ai: the presenter begun again on the monitor under the window: true where the monitor was taken; else the new
    // ai: presenter shows the window as the old one did
    private fun lease(): Boolean {
        keepPlace?.invoke()
        Native.presentStop(p); p = 0L
        val pr = Native.presentStart(canvas, h, fps, size / 100f, edge, vsync, true, canvas.place())
        if (pr == 0L) { fail("The code cannot be shown: ${Native.txError()}"); return true }
        p = pr
        if (paused) Native.presentPause(p, true)
        val st = parse(Native.presentStats(p))
        leased = st?.str("direct").orEmpty().isNotEmpty()
        leasedOn = st?.str("directName").orEmpty().ifEmpty { st?.str("direct").orEmpty() }
        if (leased) {
            // ai: the format for the monitor taken: its pixels the area (an auto pick, Size, Margin and Spacing measured on
            // ai: it), held while leased; each setting changed in the control window re-picks on it
            st?.str("directPx")?.split("x")?.mapNotNull { it.toIntOrNull() }?.takeIf { it.size == 2 && it[0] > 0 && it[1] > 0 }
                ?.let { areaW = it[0]; areaH = it[1] }
            leaseTimer.start(); canvas.repaint(); repick()
        }
        return leased
    }
    // ai: the monitor given back (a key or a click to the presenter, Stop, the window closing): full screen left and
    // ai: the window where it was before
    private fun leaseEnded() {
        leaseTimer.stop()
        leased = false
        full = false
        placeBack?.invoke()
        // ai: the area's size followed again: read now, and re-picked where the presenter goes on in the window
        if (canvas.isDisplayable) { val (w, hh) = canvas.devicePx(); areaW = w; areaH = hh }
        if (p != 0L) settle.restart()
    }
    // ai: the manager's own word on the window (its shortcut, a leave of its own, a request of ours taken): followed,
    // ai: nothing asked
    fun fullFromManager(on: Boolean) { if (!leased && on != full) { full = on; edgeRail = false } }

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
    // ai: Exclusive fullscreen's warning (Screen.kt ExclusiveTip), once: the first turn-on asks first; Cancel leaves it
    // ai: off and the warning for the next turn-on
    var askExclusive by mutableStateOf(false)
    fun answerExclusive(go: Boolean) {
        askExclusive = false
        if (go) { Prefs.exclusiveSeen = true; chooseExclusiveFull(true) }
    }
    // ai: turned on in full screen while the codes play: the monitor taken (takeOver); turned off while it is: given back
    fun chooseExclusiveFull(on: Boolean) {
        if (on && !exclusiveFull && !Prefs.exclusiveSeen) { askExclusive = true; return }
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

    // ai: the canvas's size (a resize, a window made); sending, a re-pick, or the start once it has one. While presenting
    // ai: the re-pick waits for the area to settle, 300 ms (the web's roomChanged): full screen passes through two or three
    // ai: sizes (the column folded, the manager's first size, the screen's), and a re-pick at each restarted the painters
    // ai: (2026-10-04, on a real screen: about 0.5 s of stalled presents a transition); the presenter meanwhile
    // ai: scales the picture it has into the new area.
    fun areaChanged() {
        // ai: while a monitor is taken the area is the monitor's (lease): the control window's is no measure of it (read
        // ai: again at leaseEnded)
        if (leased) return
        val (w, hh) = if (canvas.isDisplayable) canvas.devicePx() else 0 to 0
        // ai: a canvas not yet sized (its window just made) keeps the area it had: Blocks' label read "Auto, N" -> "Auto"
        // ai: -> "Auto, N" on each entry to Send (2026-11-02)
        if (w <= 0 || hh <= 0) return
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
        val pr = Native.presentStart(canvas, h, fps, size / 100f, edge, vsync, false, canvas.place())
        if (pr == 0L) { fail("The code cannot be shown: ${Native.txError()}"); return }
        p = pr
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
        if (p != 0L) { presentRaw = Native.presentStats(p); present = parse(presentRaw) }
        val e = sender?.str("error").orEmpty().ifEmpty { if (p != 0L) present?.str("error").orEmpty() else "" }
        if (e.isNotEmpty()) fail(e)
    }

    // ai: what goes, "<compressed> (<raw> raw)" once a run has made the transfer, the size alone before (the web's and the
    // ai: app's the same)
    // ai: "raw" kept with its figure (a no-break space, 2026-10-23: "(47 KB / raw)" broke there)
    private fun sizeText() = if (compressed > 0) "${Readout.bytes(compressed)} (${Readout.bytes(fileSize)}\u00a0raw)" else Readout.bytes(fileSize)

    // ai: The state line (the web's #state): what is sent, paused, or the error (red; Main.kt colours it). A line naming
    // ai: the file says so in three parts, the name between (2026-10-23), for the shared NameLine to hold its head on one
    // ai: line; any other is all `before`.
    data class StateLine(val before: String, val name: String? = null, val after: String = "")
    val stateParts: StateLine get() = when (val ph = phase) {
        is Phase.Error -> StateLine(ph.why)
        Phase.Preparing -> if (runTest) StateLine("Preparing") else StateLine("Preparing ", runName)
        Phase.On -> if (paused) StateLine("Paused") else if (runTest) StateLine("Sending") else StateLine("Sending ", runName, ", ${sizeText()}")
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
            if (blocks > 0) ", set by hand" else ", chosen from a ${room.roundToInt()} px room and nothing else"
        if (!runTest) l += "file $fileSize B${sd.num("sentBytes").toLong().let { if (it in 1 until fileSize) ", $it B as sent (zstd)" else "" }}: ${sd.num("chunks").toInt()} chunk${if (sd.num("chunks").toInt() == 1) "" else "s"}, " +
            "${sd.num("lap").toInt()} data blocks a pass, BLAKE3 ${sd.str("root").take(16)}..., header in the light" +
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

