package dev.lizard.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.movableContentOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.lizard.readout.Readout

// ai: The folds of the tool screens, kept by each app under its own keys
enum class FoldKey { RecvSettings, RecvExperimental, RecvTools, RecvCollapsed, SendSettings, SendExperimental, SendTools, SendCollapsed }

interface Folds {
    fun open(k: FoldKey): Boolean
    fun toggle(k: FoldKey)
}

// ai: The camera's permission while the app has none: the one button that gets it (the phone's; why is the state line)
class Access(val act: Act)

// ai: What Receive shows and does, the app's: the receive session's state as the app reads it, the camera's settings as
// ai: the camera offers them, the platform's picture of the camera
interface ReceiveHost {
    val folds: Folds
    val width: ColumnWidth
    val canPortrait: Boolean           // ai: an upright screen takes the portrait layout (the phone's)
    val swapInColumn: Boolean          // ai: the swap in the landscape column's bar (the desktop's; the phone's column has room for the collapser alone)
    fun goHome()
    fun goSend()
    val line: Readout.Line
    val showStats: Boolean
    val heat: String?                  // ai: the heat warning's line, null while nothing throttles
    val hint: String
    val note: String                   // ai: a filing's or a Save's outcome, in red unless it starts "Saved"
    val running: Boolean
    val starting: Boolean              // ai: Start camera inert while the camera opens
    fun toggleCamera()
    val access: Access?
    val kept: List<Act>?               // ai: the received file's actions, the first the solid Open; null while none is in
    val keptName: String
    val keptLabels: List<String>       // ai: the labels of a kept file's row, Open's first (the column's floor)
    val railRate: Double?
    val railPct: Double?
    fun cropRatio(portrait: Boolean): Float   // ai: the preview's box: the crop's width over its height
    val twoToOne: Boolean
    val off: String                    // ai: over the preview: "No camera yet", "Camera off", or "" while it shows
    val waiting: Boolean               // ai: "Getting ready" until the receiver first answers
    @Composable fun Picture(modifier: Modifier, portrait: Boolean)
    val settings: List<Setting>
    val experimental: List<Setting>    // ai: the Experimental fold's lab arms; no fold where there are none
    val tools: List<Setting>
    val about: String
}

// ai: Receive (the web's recv.html; both apps' since 2026-10-31): the camera's crop, the transfer (the actions: Open and
// ai: the platform's others once a file is in, Start camera or Stop camera; then the state line, meter, figures, the
// ai: heat warning, a filing's outcome, the hint: the desktop's order), Settings, Experimental, Developer Tools, About.
// ai: Landscape: the column on the left, its edge dragged, the camera on the right as big as the rest allows; the rail
// ai: when collapsed (play or pause, the rate, the percent, the green check and the file's actions, Home at its foot).
// ai: Portrait (a phone): the bar over one column, the camera at its top.
@Composable
fun ReceiveScreen(h: ReceiveHost) = BoxWithConstraints {
    val portrait = h.canPortrait && maxHeight > maxWidth
    val ratio = h.cropRatio(portrait)
    // ai: one preview for both layouts: a turn of the phone moves it rather than making another (a second camera view
    // ai: would end the first one's surface, and the camera with it)
    val preview = remember(h) { movableContentOf<Modifier, Boolean> { m, p -> PreviewBox(h, m, p) } }
    val swap: @Composable () -> Unit = { SwapBtn("Sender") { h.goSend() } }
    if (portrait) {
        // ai: the camera the column's width, up to 55% of the screen's height
        val tall = maxHeight * 0.55f
        ToolPortrait("Receive", { h.goHome() }, { swap() }) {
            preview(Modifier.fillMaxWidth().heightIn(max = tall), true)
            Side(h)
        }
    } else {
        val floor = maxOf(RowFloor(h.keptLabels.mapIndexed { i, l -> listOf(l) to (if (i == 0) 20.dp else 16.dp) }),
            RowFloor(listOf(listOf("Start camera", "Stop camera") to 20.dp)))
        ToolLandscape("Receive", h.folds.open(FoldKey.RecvCollapsed), { h.folds.toggle(FoldKey.RecvCollapsed) }, { h.goHome() },
            if (h.swapInColumn) swap else null, h.width, floor, rail = { RailSquares(h) }, side = { Side(h) }) {
            val w = minOf(maxHeight * ratio, maxWidth)
            preview(Modifier.width(w).height(w / ratio), false)
        }
    }
}

// ai: The rail's squares (the web's #rail): the heat's red triangle above the play while the phone throttles, play or
// ai: pause, the rate and the percent while a file or the test stream is read; once the file is kept a green check (or
// ai: the heat's triangle) and the file's actions that have an icon
@Composable
private fun RailSquares(h: ReceiveHost) {
    val kept = h.kept
    val hot = h.heat
    if (hot != null && kept == null) HeatSquare(hot)
    Square {
        IconBtn(if (h.running) Icons.pause else Icons.play, if (h.running) "Stop camera" else "Start camera",
            enabled = h.access == null && !h.starting) { h.toggleCamera() }
    }
    if (kept == null) {
        RateSquare(h.railRate)
        h.railPct?.let { PctSquare(it) }
    } else {
        if (hot != null) HeatSquare(hot)
        else Square { Icon(Icons.check, contentDescription = "Received ${h.keptName}", Modifier.size(24.dp), tint = Good) }
        for (a in kept) a.icon?.let { i -> Square { IconBtn(i, "${a.label} ${h.keptName}") { a.run() } } }
    }
}

@Composable
private fun Side(h: ReceiveHost) {
    TransferPanel(h)
    Fold("Settings", h.folds.open(FoldKey.RecvSettings), { h.folds.toggle(FoldKey.RecvSettings) }) { Settings(h.settings) }
    if (h.experimental.isNotEmpty()) Fold("Experimental", h.folds.open(FoldKey.RecvExperimental), { h.folds.toggle(FoldKey.RecvExperimental) }) { Settings(h.experimental) }
    Fold("Developer Tools", h.folds.open(FoldKey.RecvTools), { h.folds.toggle(FoldKey.RecvTools) }) { Settings(h.tools) }
    HorizontalDivider(color = Line)
    ListRow("About", h.about.ifEmpty { null })
    Spacer(Modifier.height(24.dp))
}

// ai: The transfer, the desktop's order and gaps: 12 dp under the bar the actions (once the file is kept, its actions on
// ai: one row, Open the solid one filling it, the others their own width; Start camera or Stop camera alone under them,
// ai: the full width; without the camera's permission the one button that gets it, why on the state line), 12 dp under
// ai: them the state line, the meter, the figures, then 4 dp apart the heat warning, a filing's outcome and one hint,
// ai: 16 dp before the folds. Show statistics off: no green line once a file is in.
@Composable
private fun TransferPanel(h: ReceiveHost) {
    Column(Modifier.fillMaxWidth().padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val access = h.access
        val kept = h.kept
        if (access != null) Btn(access.act.label, Kind.Primary, modifier = Modifier.widthIn(max = LocalButtonsWidest.current).fillMaxWidth()) { access.act.run() }
        else {
            if (kept != null) Row(Modifier.widthIn(max = LocalButtonsWidest.current).fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                kept.forEachIndexed { i, a ->
                    if (i == 0) Btn(a.label, Kind.Primary, modifier = Modifier.weight(1f)) { a.run() } else Btn(a.label, pad = 16.dp) { a.run() }
                }
            }
            // ai: one solid button at a time: Start camera while nothing waits, else Open
            Btn(if (h.running) "Stop camera" else "Start camera", if (!h.running && kept == null) Kind.Primary else Kind.Tonal,
                enabled = !h.starting, modifier = Modifier.widthIn(max = LocalButtonsWidest.current).fillMaxWidth()) { h.toggleCamera() }
        }
    }
    val ln = h.line
    val shown = ln.text.isNotEmpty() && !(ln.tone == Readout.Tone.Good && !h.showStats)
    val named = ln.name
    if (shown && named != null) NameLine(ln.before, named, ln.after, MaterialTheme.typography.titleMedium, toneColour(ln.tone), Modifier.padding(top = 12.dp, bottom = 2.dp))
    else if (shown) Text(ln.text, style = MaterialTheme.typography.titleMedium, color = toneColour(ln.tone), modifier = Modifier.padding(top = 12.dp, bottom = 2.dp))
    Meter(ln.frac)
    if (ln.nums.isNotEmpty()) Text(ln.nums, style = MaterialTheme.typography.bodyMedium, color = Muted)
    h.heat?.let { Warning(it, Modifier.padding(top = 4.dp)) }
    if (h.note.isNotEmpty()) Text(h.note, style = MaterialTheme.typography.bodyMedium, color = if (h.note.startsWith("Saved")) Muted else Bad, modifier = Modifier.padding(top = 4.dp))
    if (h.hint.isNotEmpty()) Text(h.hint, style = MaterialTheme.typography.bodyMedium, color = Muted, modifier = Modifier.padding(top = 4.dp))
    Spacer(Modifier.height(16.dp))
}

// ai: The camera's crop as the decoder reads it, in its box on the soft fill, square corners (the web's #view): the
// ai: platform's picture; in 2:1 a red 1 dp line across the middle of the crop's long side, where the receiver cuts the
// ai: two halves; over it while the camera is off why, with its icon, and "Getting ready" with a spinner until the
// ai: receiver first answers (Readout.gettingReady)
@Composable
private fun PreviewBox(h: ReceiveHost, modifier: Modifier, portrait: Boolean) {
    val ratio = h.cropRatio(portrait)
    Box(modifier.clipToBounds().background(Soft), contentAlignment = Alignment.Center) {
        Box(Modifier.aspectRatio(ratio, matchHeightConstraintsFirst = true).clipToBounds(), contentAlignment = Alignment.Center) {
            h.Picture(Modifier.fillMaxSize(), portrait)
            if (h.twoToOne) Canvas(Modifier.matchParentSize()) {
                val wide = size.width >= size.height
                val a = if (wide) Offset(size.width / 2, 0f) else Offset(0f, size.height / 2)
                val b = if (wide) Offset(size.width / 2, size.height) else Offset(size.width, size.height / 2)
                drawLine(Color(0xFFE53935), a, b, strokeWidth = 1.dp.toPx())
            }
            if (h.off.isNotEmpty() || h.waiting) Column(Modifier.matchParentSize().background(Soft),
                verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically), horizontalAlignment = Alignment.CenterHorizontally) {
                if (h.off.isNotEmpty()) Icon(Icons.cameraOff, contentDescription = null, Modifier.size(24.dp), tint = Muted) else Spinner()
                Text(h.off.ifEmpty { "Getting ready" }, style = MaterialTheme.typography.bodyMedium, color = Muted, textAlign = TextAlign.Center)
            }
        }
    }
}

// ai: The heat warning (the phone's): the red triangle and its line; in a rail the triangle alone, its line the
// ai: description
@Composable
fun Warning(text: String, modifier: Modifier = Modifier) {
    Row(modifier.semantics(mergeDescendants = true) {}, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Icon(Icons.alert, contentDescription = "Warning", Modifier.size(20.dp), tint = Bad)
        Text(text, style = MaterialTheme.typography.bodyMedium, color = Bad)
    }
}

@Composable
fun HeatSquare(text: String) = Square { Icon(Icons.alert, contentDescription = text, Modifier.size(24.dp), tint = Bad) }
