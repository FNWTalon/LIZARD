package dev.lizard.desktop

import androidx.compose.foundation.VerticalScrollbar
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

// ai: The sender's window, the web's page in landscape (lizard-web/send.html): a column on the left (the web's 20rem;
// ai: its collapser folds it to a 56 dp rail with Pause or Play and the code's capacity, as the web's rail), a hairline,
// ai: and the code area on the right, white, the presenter's canvas. Full screen folds the column to the rail.
@Composable
fun SendScreen(s: SendState, window: ComposeWindow) {
    MaterialTheme(colorScheme = LizardColours, typography = LizardType) {
        Row(Modifier.fillMaxSize().background(Bg)) {
            if (s.railShown) Rail(onClick = { s.toggleRail() }) {
                val on = s.phase == SendState.Phase.On
                Square {
                    IconBtn(if (on && !s.paused) Icons.pause else Icons.play, if (!on) "Start" else if (s.paused) "Resume" else "Pause",
                        enabled = on || s.canStart) { if (on) s.pause(!s.paused) else s.requestStart() }
                }
                RateSquare(s.capacityKBs)
            } else Column(Modifier.width(320.dp).fillMaxHeight()) {
                Box(Modifier.padding(horizontal = 16.dp)) { TopBar("Send") { CollapseBtn(false) { s.toggleRail() } } }
                Box(Modifier.weight(1f).fillMaxWidth()) {
                    val scroll = rememberScrollState()
                    Column(Modifier.fillMaxSize().verticalScroll(scroll).padding(horizontal = 16.dp)) { Side(s, window) }
                    VerticalScrollbar(rememberScrollbarAdapter(scroll), Modifier.align(Alignment.CenterEnd).fillMaxHeight())
                }
            }
            VerticalDivider(color = Line)
            SwingPanel(background = Bg, factory = { s.canvas }, modifier = Modifier.weight(1f).fillMaxHeight())
        }
        if (s.askBrightness) BrightnessTip(s, window)
    }
}

private const val TIP = "To maximize efficiency, increase the contrast and brightness of your display."

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
    LaunchedEffect(s.file, s.test, s.phase, s.full) { s.canvas.idleLook() }
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
    s.stateLine.let { if (it.isNotEmpty()) Text(it, style = MaterialTheme.typography.titleMedium, color = if (bad) Bad else Fg, modifier = Modifier.padding(top = 12.dp)) }
    s.figures.let { if (it.isNotEmpty()) Text(it, style = MaterialTheme.typography.bodyMedium, color = Muted, modifier = Modifier.padding(top = 2.dp)) }
    Spacer(Modifier.height(16.dp))
    Fold("Settings", s.settingsOpen, { s.toggleSettings() }) {
        Fields {
            // ai: GPU and CPU alone (2026-10-05; an Auto chip until then): with neither chosen the encoder is auto's and
            // ai: the chip of the painter running is marked; a click chooses that one outright
            Field("Encoder") {
                Chips {
                    Chip("GPU", s.enc == "gpu" || (s.enc == "auto" && s.painterNow == "gpu")) { s.chooseEnc("gpu") }
                    Chip("CPU", s.enc == "cpu" || (s.enc == "auto" && s.painterNow == "cpu")) { s.chooseEnc("cpu") }
                }
            }
            Group("Code") {
                BlocksField(s)
                Field("FPS", "${s.fps}") { Bar(s.fps.toFloat(), 1f..60f, 58, done = { s.commitFps() }) { s.chooseFps(it.roundToInt()) } }
                Field("Size", "${s.size}%") { Bar(s.size.toFloat(), 25f..100f, 74, done = { s.commitSize() }) { s.chooseSize(it.roundToInt()) } }
                if (s.codes == 2) Field("Gap", "${s.gap} modules") { Bar(s.gap.toFloat(), 0f..64f, 63, done = { s.repick() }) { s.chooseGap(it.roundToInt()) } }
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
// ai: title until then), its title "Auto, 60 blocks, 28.1 KB" once auto's pick in the area is known, every step right of
// ai: it a count by hand, committed on release.
@Composable
private fun BlocksField(s: SendState) {
    val auto = s.blocks == 0
    val b = if (auto) s.autoBlocks else s.blocks
    // ai: under a rate profile (LIZ_TIERS, 2026-10-07) the slider is not read: the profile and the blocks it paints
    val profile = System.getenv("LIZ_TIERS").orEmpty()
    Field("Blocks", if (s.tiers.isNotEmpty()) "${s.tiers}: ${Fmt.blocks(s.sentBlocks)}" else if (profile.isNotEmpty()) profile
        else if (auto) (if (b > 0) "Auto, ${Fmt.blocks(b)}" else "Auto") else Fmt.blocks(b)) {
        Bar(s.blocks.toFloat(), 0f..128f, 127, done = { s.repick() }) { s.chooseBlocks(it.roundToInt()) }
    }
}

