package dev.lizard.ui

import androidx.compose.ui.draw.clip
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
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.text.style.TextOverflow
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
    val note: String                   // ai: a filing's or a Save's failure; "" none
    val running: Boolean
    val starting: Boolean              // ai: Start inert while the camera opens
    fun toggleCamera()
    val access: Access?
    val kept: List<Act>?               // ai: the received file's actions, the first the solid Open (each with its icon: a tile's
                                       // ai: and the rail's); null while none is in
    val keptName: String
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
// ai: the platform's others once a file is in, Start or Stop (the camera); then the state line, meter, figures, the
// ai: heat warning, the hint: the desktop's order; a filing's failure is the preview's to say), Settings, Experimental, Developer Tools, About.
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
    // ai: the received file in the camera's place once it is in and the camera is off (2026-11-06, the web's #done)
    val got = h.kept != null && !h.running && !h.starting && errorOf(h) == null
    if (portrait) {
        // ai: the camera the column's width, up to 55% of the screen's height
        val tall = maxHeight * 0.55f
        ToolPortrait("Receive", { h.goHome() }, { swap() }) {
            if (got) ReceivedPanel(h, Modifier.fillMaxWidth(), column = null) else preview(Modifier.fillMaxWidth().heightIn(max = tall), true)
            Side(h)
        }
    } else {
        // ai: the column's floor: Start or Stop, its one button (the received file's actions are the camera's area's)
        val floor = RowFloor(listOf(listOf("Start", "Stop") to 20.dp))
        ToolLandscape("Receive", h.folds.open(FoldKey.RecvCollapsed), { h.folds.toggle(FoldKey.RecvCollapsed) }, { h.goHome() },
            if (h.swapInColumn) swap else null, h.width, floor, rail = { RailSquares(h) }, side = { Side(h) }) {
            if (got) ReceivedPanel(h, Modifier.fillMaxSize(), column = 576.dp)
            else {
                val w = minOf(maxHeight * ratio, maxWidth)
                preview(Modifier.width(w).height(w / ratio), false)
            }
        }
    }
}

// ai: The received file (2026-11-06; the web's recv.html #done), in the camera's place once the camera is off, in the
// ai: sender's upload geometry (Send.kt Upload): its actions as tiles in one row in equal
// ai: cells, Open the solid one, then a card, "Received" and how it went, the file as an item tile. `column`: in
// ai: landscape a column at most that wide centred in the camera's area (the sender's 576 dp); null, the column's width
// ai: at the top (portrait, a phone's: the tiles across it).
@Composable
private fun ReceivedPanel(h: ReceiveHost, modifier: Modifier, column: Dp?) {
    val acts = h.kept ?: return
    val ln = h.line
    BoxWithConstraints(modifier) {
        val w = if (column != null) minOf(maxWidth - 32.dp, column) else maxWidth
        Column(if (column != null) Modifier.fillMaxSize() else Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = if (column != null) Arrangement.spacedBy(16.dp, Alignment.CenterVertically) else Arrangement.spacedBy(16.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                acts.forEachIndexed { i, a -> Tile(a.icon ?: Icons.file, a.label, (w - 12.dp * (acts.size - 1)) / acts.size, solid = i == 0, onClick = a.run) }
            }
            Column(Modifier.width(w).clip(RoundedCornerShape(12.dp)).background(Card).padding(start = 12.dp, end = 12.dp, top = 4.dp, bottom = 12.dp)) {
                // ai: the title alone on its line, how it went under it (2026-11-06: beside it, it took the row and
                // ai: squeezed the title to a letter a line); the figures only with Show statistics on, else the title
                // ai: counts what came, "Received 10 files", "Received 1 file" (the size is the tile's)
                val good = ln.tone == Readout.Tone.Good
                Column(Modifier.fillMaxWidth().padding(start = 4.dp, end = 4.dp, top = 8.dp, bottom = 12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(if (good && !h.showStats && ln.files > 0) "Received ${countOf(ln.files)}" else "Received",
                        style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = Fg)
                    if (good && h.showStats) Text(ln.after.removePrefix(", "), style = MaterialTheme.typography.bodyMedium, color = Muted)
                }
                ItemTile(h.keptName, if (ln.tone == Readout.Tone.Good) ln.size else "", Icons.file, TILE.dp, onRemove = null)
            }
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
    val err = errorOf(h)
    if (err != null) AlertSquare(err) else if (hot != null && kept == null) AlertSquare(hot)
    Square {
        IconBtn(if (h.running) Icons.pause else Icons.play, if (h.running) "Stop" else "Start",
            enabled = h.access == null && !h.starting) { h.toggleCamera() }
    }
    if (kept == null) {
        RateSquare(h.railRate)
        h.railPct?.let { PctSquare(it) }
    } else {
        if (hot != null && err == null) AlertSquare(hot)
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
    AboutFold(h.about)
    SourceRow()
    Spacer(Modifier.height(24.dp))
}

// ai: The transfer, the desktop's order and gaps: 12 dp under the bar the actions (once the file is kept, its actions on
// ai: one row, Open the solid one filling it, the others their own width; Start or Stop alone under them,
// ai: the full width; on the desktop every button its label's width, LocalButtonsFill; without the camera's permission
// ai: the one button that gets it, why on the state line), 12 dp under
// ai: them the state line, the meter, the figures, then 4 dp apart the heat warning and one hint (a filing's failure over the preview),
// ai: 16 dp before the folds. Show statistics off: no green line once a file is in.
@Composable
private fun TransferPanel(h: ReceiveHost) {
    Column(Modifier.fillMaxWidth().padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val access = h.access
        val kept = h.kept
        val wide = if (LocalButtonsFill.current) Modifier.fillMaxWidth() else Modifier
        if (access != null) Btn(access.act.label, Kind.Primary, modifier = wide) { access.act.run() }
        else {
            // ai: one solid button at a time: Start while nothing waits, else Open (the camera's area's, ReceivedPanel)
            Btn(if (h.running) "Stop" else "Start", if (!h.running && kept == null) Kind.Primary else Kind.Tonal,
                enabled = !h.starting, modifier = wide) { h.toggleCamera() }
        }
    }
    val ln = h.line
    // ai: an error is the preview's to say (errorOf), not the column's
    // ai: the received line is the card's to say (ReceivedPanel, 2026-11-06), never the column's; Show statistics off hid
    // ai: it before
    val shown = ln.text.isNotEmpty() && ln.tone != Readout.Tone.Good && ln.tone != Readout.Tone.Bad
    val named = ln.name
    // ai: Within a run (Start until it stops) a line once shown keeps its room until the run ends, drawn clear
    // ai: when it falls silent (2026-11-02: Receiving, its meter and figures, and the hint each moved the folds under them
    // ai: as they came and went); none is held before it has had words, so a camera starting shows no gap (a run's whole
    // ai: room held from Start stood empty under Stop). The run's file in ends it at once, the kept row and the
    // ai: lines' going in one frame (the camera stops a poll later); a run started over a file already kept holds on.
    // ai: Every line is one line.
    val keptAtStart = remember(h.running) { h.kept != null }
    val run = h.running && (h.kept == null || keptAtStart)
    val last = remember(h.running) { arrayOf("", "", "") }   // ai: the line, the figures and the hint as last shown this run
    val meterSeen = remember(h.running) { BooleanArray(1) }
    if (run && ln.frac != null) meterSeen[0] = true
    fun said(i: Int, now: String): String { if (run && now.isNotEmpty()) last[i] = now; return now }
    val lineNow = said(0, if (shown) ln.text else "")
    if (shown && named != null) NameLine(ln.before, named, ln.after, MaterialTheme.typography.titleMedium, toneColour(ln.tone), Modifier.padding(top = 12.dp, bottom = 2.dp))
    else if (lineNow.isNotEmpty() || (run && last[0].isNotEmpty())) Text(lineNow.ifEmpty { last[0] }, style = MaterialTheme.typography.titleMedium,
        color = if (lineNow.isEmpty()) Color.Transparent else toneColour(ln.tone), maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 12.dp, bottom = 2.dp))
    Meter(ln.frac.takeIf { ln.tone != Readout.Tone.Good }, hold = run && meterSeen[0])
    val numsNow = said(1, ln.nums)
    if (numsNow.isNotEmpty() || (run && last[1].isNotEmpty())) Text(numsNow.ifEmpty { last[1] }, style = MaterialTheme.typography.bodyMedium,
        color = if (numsNow.isEmpty()) Color.Transparent else Muted, maxLines = 1, overflow = TextOverflow.Ellipsis)
    h.heat?.let { Warning(it, Modifier.padding(top = 4.dp)) }
    // ai: the hint is the running camera's: held still while it runs (steady), its room kept as the lines' are, gone with it
    val hint = said(2, steady(h.hint, on = 0).takeIf { run }.orEmpty())
    if (hint.isNotEmpty() || (run && last[2].isNotEmpty())) Text(hint.ifEmpty { last[2] }, style = MaterialTheme.typography.bodyMedium,
        color = if (hint.isEmpty()) Color.Transparent else Muted, modifier = Modifier.padding(top = 4.dp))
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
            val err = errorOf(h)
            if (err != null || h.off.isNotEmpty() || h.waiting) Column(Modifier.matchParentSize().background(Soft).padding(horizontal = 24.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically), horizontalAlignment = Alignment.CenterHorizontally) {
                when {
                    err != null -> Icon(Icons.alert, contentDescription = null, Modifier.size(24.dp), tint = Bad)
                    h.off.isNotEmpty() -> Icon(Icons.cameraOff, contentDescription = null, Modifier.size(24.dp), tint = Muted)
                    else -> Spinner()
                }
                Text(err ?: h.off.ifEmpty { "Getting ready" }, style = MaterialTheme.typography.bodyMedium, color = Muted, textAlign = TextAlign.Center)
            }
        }
    }
}

// ai: What went wrong (2026-11-02), said over the preview, its cover the red triangle with the words under it, never in
// ai: the column (the collapsed rail's AlertSquare at most): the line's error (the phase's why, "Reading stopped. ...")
// ai: or a filing's refusal (the note)
private fun errorOf(h: ReceiveHost): String? = h.line.takeIf { it.tone == Readout.Tone.Bad }?.text?.takeIf { it.isNotEmpty() }
    ?: h.note.takeIf { it.isNotEmpty() }

// ai: The heat warning (the phone's): the red triangle and its line; in a rail the triangle alone, its line the
// ai: description
@Composable
fun Warning(text: String, modifier: Modifier = Modifier) {
    Row(modifier.semantics(mergeDescendants = true) {}, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Icon(Icons.alert, contentDescription = "Warning", Modifier.size(20.dp), tint = Bad)
        Text(text, style = MaterialTheme.typography.bodyMedium, color = Bad)
    }
}

// ai: the red triangle in a rail's square, its words the description: the phone's heat, a receiver's error
@Composable
fun AlertSquare(text: String) = Square { Icon(Icons.alert, contentDescription = text, Modifier.size(24.dp), tint = Bad) }
