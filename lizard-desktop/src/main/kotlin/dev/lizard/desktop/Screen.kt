@file:OptIn(ExperimentalFoundationApi::class, ExperimentalComposeUiApi::class)

package dev.lizard.desktop

import dev.lizard.pick.Pick
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.draganddrop.dragAndDropTarget
import androidx.compose.runtime.remember
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.draganddrop.DragAndDropEvent
import androidx.compose.ui.draganddrop.DragAndDropTarget
import androidx.compose.ui.draganddrop.awtTransferable
import java.awt.datatransfer.DataFlavor
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.awt.ComposeWindow
import androidx.compose.ui.awt.SwingPanel
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogWindow
import androidx.compose.ui.window.WindowPosition
import androidx.compose.ui.window.rememberDialogState
import kotlin.math.roundToInt
import kotlinx.coroutines.delay
import androidx.compose.ui.awt.ComposeDialog
import java.awt.Dialog
import java.awt.MouseInfo
import java.awt.Window

// ai: The sender's window, the web's page in landscape (lizard-web/send.html): a column on the left (the web's 20rem;
// ai: its collapser folds it to a 56 dp rail with Pause or Play and the code's capacity, as the web's rail), a hairline,
// ai: and the code area on the right, white, the presenter's canvas. Full screen shows the code area alone. Since the
// ai: app's screens (2026-10-25, App.kt): the bar's swap to the receiver (the web's #swap, "Receiver"), and Home at the
// ai: rail's foot (the web's: Home from the collapsed rail's last square alone in landscape).
@Composable
fun SendScreen(s: SendState, window: ComposeWindow, onSwap: () -> Unit = {}, onHome: () -> Unit = {}) {
    MaterialTheme(colorScheme = LizardColours, typography = LizardType) {
        // ai: files and folders dropped on the column are added as on the code area (SendState.dropped), the area's
        // ai: outline shown while a drag is over the window (2026-10-23)
        val drops = remember { object : DragAndDropTarget {
            override fun onEntered(event: DragAndDropEvent) = s.dragging(true)
            override fun onExited(event: DragAndDropEvent) = s.dragging(false)
            override fun onEnded(event: DragAndDropEvent) = s.dragging(false)
            override fun onDrop(event: DragAndDropEvent): Boolean { s.dragging(false); return s.dropped(event.awtTransferable) }
        } }
        Row(Modifier.fillMaxSize().background(Bg).dragAndDropTarget({ it.awtTransferable.isDataFlavorSupported(DataFlavor.javaFileListFlavor) }, drops)) {
            // ai: full screen: the code alone (2026-10-23, as the web's and the app's; the rail beside it before), the rail
            // ai: over its left edge while the pointer is near it (EdgeRail)
            if (!s.full) {
                if (s.railShown) Rail(onClick = { s.toggleRail() }, foot = { Square { IconBtn(Icons.back, "Home", onClick = onHome) } }) { RailSquares(s) }
                else Column(Modifier.width(320.dp).fillMaxHeight()) {
                    Box(Modifier.padding(horizontal = 16.dp)) { TopBar("Send") { SwapBtn("Receiver", onSwap); CollapseBtn(false) { s.toggleRail() } } }
                    Box(Modifier.weight(1f).fillMaxWidth()) {
                        val scroll = rememberScrollState()
                        Column(Modifier.fillMaxSize().verticalScroll(scroll).padding(horizontal = 16.dp)) { Side(s, window) }
                        VerticalScrollbar(rememberScrollbarAdapter(scroll), Modifier.align(Alignment.CenterEnd).fillMaxHeight())
                    }
                }
                VerticalDivider(color = Line)
            }
            SwingPanel(background = Bg, factory = { s.canvas }, modifier = Modifier.weight(1f).fillMaxHeight())
        }
        if (s.full) EdgeRail(s, window)
        if (s.askBrightness) BrightnessTip(s, window)
    }
}

// ai: the collapsed column's squares: Start, Pause or Resume, and what the code offers
@Composable
private fun RailSquares(s: SendState) {
    val on = s.phase == SendState.Phase.On
    Square {
        IconBtn(if (on && !s.paused) Icons.pause else Icons.play, if (!on) "Start" else if (s.paused) "Resume" else "Pause",
            enabled = on || s.canStart) { s.playPause() }
    }
    RateSquare(s.capacityKBs)
}

// ai: Full screen's rail (2026-10-23): the collapsed column over the screen's left edge while the pointer is within
// ai: EDGE_NEAR of it (SendState.CodeCanvas) or over the rail, in a window of its own (nothing of Compose draws over the
// ai: code's heavyweight canvas), over the code rather than beside it so the area and the format stay as they are. A
// ai: popup (override-redirect on X11): the window manager neither places it (a dialog it centres on its owner at
// ai: first) nor shades it, nor gives it the focus full screen holds. Mapped for as long as full screen lasts, parked
// ai: off every screen while hidden, so a show is a move (a window mapped and drawn at each show took 0.5 to 1.1 s).
// ai: Its collapser leaves full screen with the column open. Hidden once the pointer is off it (read a few times a
// ai: second: the canvas below hears nothing while the pointer is over the rail, nor once it goes past the screen's edge).
@Composable
private fun EdgeRail(s: SendState, owner: ComposeWindow) {
    val shown = s.edgeRail
    DialogWindow(visible = true,
        create = { ComposeDialog(owner, Dialog.ModalityType.MODELESS).apply { type = Window.Type.POPUP; isUndecorated = true; isResizable = false; focusableWindowState = false } },
        dispose = ComposeDialog::dispose,
        update = { d ->
            val h = maxOf(1, s.canvas.height)
            if (shown && s.canvas.isShowing) s.canvas.locationOnScreen.let { d.setBounds(it.x, it.y, 57, h) } else d.setBounds(-10000, -10000, 57, h)
        }) {
        LaunchedEffect(shown) {
            while (shown) {
                delay(150)
                val p = MouseInfo.getPointerInfo()?.location
                if (p == null || !window.bounds.contains(p)) { s.edgeRail = false; break }
            }
        }
        Row(Modifier.fillMaxSize().background(Bg)) {
            Rail(onClick = { s.toggleFull(); if (s.collapsed) s.toggleRail() }) { RailSquares(s) }
            VerticalDivider(color = Line)
        }
    }
}

// ai: power saving since 2026-10-23 (a first-time tip, not a warning: a power profile is no sure sign of dropped pictures)
private const val TIP = "To maximize efficiency, increase the contrast and brightness of your display. Turn off power saving: it can make pictures drop."

// ai: The brightness tip before the first send (SendState.requestStart; send.html #first, its words): a modal window of
// ai: its own, centred on the sender's, as tall as its text (a dialog drawn in the sender's window sits under the code
// ai: area's canvas, a native window above Compose's). Escape cancels it (Main.kt).
@Composable
private fun BrightnessTip(s: SendState, owner: ComposeWindow) {
    val style = MaterialTheme.typography.bodyLarge
    val measurer = rememberTextMeasurer()
    val w = 400.dp
    val textH = with(LocalDensity.current) { measurer.measure(TIP, style, constraints = Constraints(maxWidth = (w - 40.dp).roundToPx())).size.height.toDp() }
    val h = 20.dp + textH + 16.dp + 40.dp + 16.dp
    val state = rememberDialogState(WindowPosition(owner.x.dp + (owner.width.dp - w) / 2, owner.y.dp + (owner.height.dp - h) / 2), DpSize(w, h))
    DialogWindow({ s.answerBrightness(false) }, state, title = "LIZARD sender", undecorated = true, resizable = false) {
        Column(Modifier.fillMaxSize().background(Bg).border(1.dp, Line).padding(start = 20.dp, top = 20.dp, end = 20.dp, bottom = 16.dp)) {
            Text(TIP, style = style)
            Spacer(Modifier.weight(1f))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
                Btn("Cancel", Kind.Text) { s.answerBrightness(false) }
                Btn("Start", Kind.Primary) { s.answerBrightness(true) }
            }
        }
    }
}

// ai: The column (send.html #side and #settings): Start (Stop, and Pause or Resume, while sending) beside Fullscreen,
// ai: the state line (the file's name and size while idle, as the web's) and the figures, then Settings (the encoder
// ai: and the code), Developer Tools (what is sent, the readout) and About. Labels bare, no descriptions (2026-10-02).
// ai: The file is chosen in the code area (SendState.kt CodeCanvas, 2026-10-07; a row here with Change until then).
@Composable
private fun ColumnScope.Side(s: SendState, window: ComposeWindow) {
    val idle = !s.sending
    // ai: the code area's idle look follows the file, the payload and the run
    LaunchedEffect(s.choice, s.reading, s.test, s.phase, s.full, s.items.size) { s.canvas.idleLook() }
    Spacer(Modifier.height(12.dp))
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        if (idle) Btn("Start", Kind.Primary, enabled = s.canStart, modifier = Modifier.weight(1f)) { s.requestStart() }
        else {
            Btn("Stop", modifier = Modifier.weight(1f), pad = 16.dp) { s.stop() }
            if (s.paused) Btn("Resume", Kind.Primary, pad = 16.dp) { s.pause(false) }
            else Btn("Pause", enabled = s.phase == SendState.Phase.On, pad = 16.dp) { s.pause(true) }
        }
        Btn("Fullscreen", pad = 16.dp) { s.toggleFull() }
    }
    val bad = s.phase is SendState.Phase.Error || (s.phase == SendState.Phase.Idle && s.loadError.isNotEmpty())
    // ai: a line naming the file holds its head on one line (Parts.kt NameLine, 2026-10-23: a long name wrapped over two)
    val st = s.stateParts
    if (st.name != null) NameLine(st.before, st.name, st.after, MaterialTheme.typography.titleMedium, if (bad) Bad else Fg, Modifier.padding(top = 12.dp))
    else if (st.before.isNotEmpty()) Text(st.before, style = MaterialTheme.typography.titleMedium, color = if (bad) Bad else Fg, modifier = Modifier.padding(top = 12.dp))
    s.figures.let { if (it.isNotEmpty()) Text(it, style = MaterialTheme.typography.bodyMedium, color = Muted, modifier = Modifier.padding(top = 2.dp)) }
    Spacer(Modifier.height(16.dp))
    Fold("Settings", s.settingsOpen, { s.toggleSettings() }) {
        Fields {
            // ai: two groups (2026-10-23): Code, what the symbol carries, then Display, how this machine paints and shows it
            Group("Code") {
                BlocksField(s)
                if (s.vsync) Field("FPS") { Chips { for (v in Pick.FPS) Chip("$v", s.fps == v) { s.chooseFps(v); s.commitFps() } } }
                else Field("FPS", "${s.fps}") { Bar(s.fps.toFloat(), 1f..60f, 58, done = { s.commitFps() }) { s.chooseFps(it.roundToInt()) } }
                // ai: the rings alone (2026-10-05; an Auto chip, the default ring, until then): none chosen marks the default
                Field("Ring") {
                    Chips {
                        for (i in Pick.RINGS.indices) Chip("${Pick.RINGS[i]}", s.ring == i || (s.ring == -1 && i == Pick.RING_DEFAULT)) { s.chooseRing(i) }
                    }
                }
                // ai: two codes side by side for the app's 2:1 crop (2026-09-28; send.html #codes); "Two" alone since 2026-10-05,
                // ai: its "(browsers cannot decode 2 codes)" wider than the row
                Field("Codes") {
                    Chips {
                        Chip("One", s.codes == 1) { s.chooseCodes(1) }
                        Chip("Two", s.codes == 2) { s.chooseCodes(2) }
                    }
                }
            }
            Group("Display") {
                // ai: GPU and CPU alone (2026-10-05; an Auto chip until then): with neither chosen the encoder is auto's and
                // ai: the chip of the painter running is marked; a click chooses that one outright
                Field("Encoder") {
                    Chips {
                        Chip("GPU", s.enc == "gpu" || (s.enc == "auto" && s.painterNow == "gpu")) { s.chooseEnc("gpu") }
                        Chip("CPU", s.enc == "cpu" || (s.enc == "auto" && s.painterNow == "cpu")) { s.chooseEnc("cpu") }
                    }
                }
                Field("VSync") {
                    Chips {
                        Chip("On", s.vsync) { s.chooseVsync(true) }
                        Chip("Off", !s.vsync) { s.chooseVsync(false) }
                    }
                }
                Field("Size", "${s.size}%") { Bar(s.size.toFloat(), 25f..100f, 74, done = { s.commitSize() }) { s.chooseSize(it.roundToInt()) } }
                if (s.codes == 2) Field("Spacing", "${s.spacing} modules") {
                    Bar(s.spacing.toFloat(), Room.SPACING_MIN.toFloat()..Room.SPACING_MAX.toFloat(), Room.SPACING_MAX - Room.SPACING_MIN - 1, done = { s.repick() }) { s.chooseSpacing(it.roundToInt()) }
                }
                Field("Margin", "${s.margin} modules") {
                    Bar(s.margin.toFloat(), Pick.QUIET.toFloat()..Room.MARGIN_MAX.toFloat(), Room.MARGIN_MAX - Pick.QUIET - 1, done = { s.repick() }) { s.chooseMargin(it.roundToInt()) }
                }
            }
        }
    }
    Fold("Developer Tools", s.toolsOpen, { s.toggleTools() }) {
        Fields {
            Field("Payload") {
                Chips {
                    Chip("File", !s.test, enabled = idle) { s.chooseTest(false) }
                    Chip("Test stream", s.test, enabled = idle) { s.chooseTest(true) }
                }
            }
        }
        s.readout.let { if (it.isNotEmpty()) CodeBlock(it) }
    }
    HorizontalDivider(color = Line)
    ListRow("About", s.about.ifEmpty { null })
    Spacer(Modifier.height(24.dp))
}

// ai: Blocks a frame (send.html #blocks), 0 for auto: the slider's first step is auto (2026-10-05; an Auto chip over the
// ai: title until then), its value "Auto, 60, 28.1 KB" once auto's pick in the area is known, every step right of
// ai: it a count by hand from 2 (LIZARD-16) to 128, committed on release.
@Composable
private fun BlocksField(s: SendState) {
    val auto = s.blocks == 0
    // ai: the slider sets the format's size (its sub-channels / 8); the title is the blocks that size carries under the
    // ai: format's rate profile, and their bytes (2026-10-07; LIZARD-568, the slider's 71, carries 67)
    val v = if (auto) s.autoBlocks else s.blocks
    val b = if (v > 0) Native.blocksFor(8 * v) else 0
    Field("Blocks", if (auto) (if (b > 0) "Auto, ${Fmt.blocks(b)}" else "Auto") else Fmt.blocks(b)) {
        Bar(if (auto) 1f else s.blocks.toFloat(), 1f..128f, 126, done = { s.repick() }) { val n = it.roundToInt(); s.chooseBlocks(if (n < 2) 0 else n) }
    }
}

