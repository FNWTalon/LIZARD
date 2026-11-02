package dev.lizard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.displayCutout
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.movableContentOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import dev.lizard.readout.Readout
import kotlinx.coroutines.delay

// ai: The state line a send shows: its head, the file's name (its stem cut, its ending kept: NameLine) and the rest;
// ai: in red for an error
class SendLine(val before: String, val name: String?, val after: String, val bad: Boolean)

// ai: What Send shows and does, the app's: the run (the sender's state), its settings, the platform's code area and
// ai: the windows and full screen around it
interface SendHost {
    val folds: Folds
    val width: ColumnWidth
    val canPortrait: Boolean
    val swapInColumn: Boolean
    fun goHome()
    fun goReceive()
    val busy: Boolean                  // ai: a send runs or is being prepared
    val on: Boolean                    // ai: a send runs
    val paused: Boolean
    val canStart: Boolean
    fun requestStart()
    fun stop()
    fun pause(on: Boolean)
    val pauseShown: Boolean            // ai: Pause or Resume beside Stop (the desktop's while sending; the phone's paused)
    val fullscreen: Boolean            // ai: a Fullscreen button (the desktop's; the phone's send is the whole screen)
    fun toggleFull()
    val line: SendLine
    val figures: String
    val heat: String?
    val railCapacity: Double?
    val settings: List<Setting>
    val experimental: List<Setting>    // ai: the Experimental fold's lab arms; no fold where there are none
    val tools: List<Setting>
    val readout: Code?                 // ai: the lab's text under Developer Tools' fields (CodeBlock's own 6 dp under them)
    val about: String
    val codeAlone: Boolean             // ai: the code alone on the screen (the phone's send running; the desktop's full screen)
    // ai: the code alone drawn as the landscape layout bare (ToolLandscape's `bare`), the code area never moved in the tree:
    // ai: the desktop's, whose code area is a heavyweight canvas a move destroys and makes again, its presenter stopped
    // ai: and begun anew (2026-11-02, every full-screen switch); the phone's CodeAlone otherwise
    val codeAloneInPlace: Boolean get() = false
    val edgeTap: Boolean               // ai: a tap near the left edge shows the rail over the code alone (the phone's)
    fun leaveCodeAlone()               // ai: the rail's collapser over the code alone
    val codePadding: Dp                // ai: the light round the code area in the page (the phone's 16 dp; the desktop's canvas fills)
    fun codeSize(maxWidth: Dp, maxHeight: Dp, portrait: Boolean): DpSize?   // ai: the code area's size in its room, null to fill it
    val rootModifier: Modifier         // ai: the platform's on the whole screen (the desktop's drops)
    @Composable fun CodeArea(modifier: Modifier)
    @Composable fun Windows()          // ai: what the platform shows apart: the desktop's tips and edge rail, the phone's question
}

// ai: Send (the web's send.html; both apps' since 2026-10-31): a column (Start, or Stop and Pause or Resume while sending,
// ai: beside Fullscreen where the platform has one; the state line naming the file; the figures; the heat warning;
// ai: Settings in two groups, Experimental, Developer Tools, About) and the code area. Landscape: the column on the left,
// ai: its edge dragged, the code area on the right; collapsed, the rail (play or pause, the capacity, Home at its foot).
// ai: Portrait (a phone): the bar over one column, the code at its top. The code alone: the code area the whole screen.
@Composable
fun SendScreen(h: SendHost) = Box(Modifier.fillMaxSize().then(h.rootModifier)) {
    // ai: one code area for every layout: a turn of the phone moves its surface rather than making another
    val code = remember(h) { movableContentOf<Modifier> { m -> h.CodeArea(m) } }
    BoxWithConstraints {
        val portrait = h.canPortrait && maxHeight > maxWidth
        if (h.codeAlone && !(h.codeAloneInPlace && !portrait)) CodeAlone(h, code)
        else if (portrait) ToolPortrait("Send", { h.goHome() }, { SwapBtn("Receiver") { h.goReceive() } }) {
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                val size = h.codeSize(maxWidth, maxHeight, true)
                code(if (size != null) Modifier.size(size.width, size.height) else Modifier.fillMaxWidth())
            }
            Side(h)
        } else ToolLandscape("Send", h.folds.open(FoldKey.SendCollapsed), { h.folds.toggle(FoldKey.SendCollapsed) }, { h.goHome() },
            if (h.swapInColumn) ({ SwapBtn("Receiver") { h.goReceive() } }) else null, h.width,
            RowFloor(listOfNotNull(listOf("Stop") to 16.dp, listOf("Pause", "Resume") to 16.dp, if (h.fullscreen) listOf("Fullscreen") to 16.dp else null)),
            mainPadding = h.codePadding, bare = h.codeAlone,
            rail = { SendRail(h) }, side = { Side(h) }) {
            val size = h.codeSize(maxWidth, maxHeight, false)
            code(if (size != null) Modifier.size(size.width, size.height) else Modifier.fillMaxSize())
        }
    }
    h.Windows()
}

// ai: The code alone (2026-10-26 on the phone, as the desktop's full screen): on the page's white, as large as the screen
// ai: allows, clear of a display's cutout; where the platform taps (the phone), a tap within EDGE of the left edge shows
// ai: the collapsed column over the code (its collapser, Pause or Resume, the rate, Home), a tap elsewhere or RAIL_MS
// ai: untouched hides it
@Composable
private fun CodeAlone(h: SendHost, code: @Composable (Modifier) -> Unit) {
    var rail by remember { mutableStateOf(false) }
    var touched by remember { mutableStateOf(0) }
    LaunchedEffect(rail, touched) { if (rail) { delay(RAIL_MS); rail = false } }
    BoxWithConstraints(Modifier.fillMaxSize().background(Bg).then(if (h.edgeTap) Modifier.pointerInput(Unit) {
        detectTapGestures { at -> rail = at.x < EDGE.toPx(); touched++ }
    }.windowInsetsPadding(WindowInsets.displayCutout) else Modifier), contentAlignment = Alignment.Center) {
        val size = h.codeSize(maxWidth, maxHeight, false)
        code(if (size != null) Modifier.size(size.width, size.height) else Modifier.fillMaxSize())
        if (rail) Row(Modifier.align(Alignment.CenterStart).fillMaxHeight().background(Bg).pointerInput(Unit) { detectTapGestures { touched++ } }) {
            Rail(onClick = { rail = false; h.leaveCodeAlone() }, foot = { Square { IconBtn(Icons.back, "Home") { h.goHome() } } }) { SendRail(h) }
            VerticalDivider(color = Line)
        }
    }
}

// ai: how near the left edge a tap shows the rail over the code alone, and how long it stays untouched
private val EDGE = 72.dp
private const val RAIL_MS = 4000L

// ai: The rail's squares: the heat's red triangle while the phone throttles, play to start (or to resume), pause while
// ai: sending, and the code's capacity
@Composable
fun SendRail(h: SendHost) {
    h.heat?.let { AlertSquare(it) }
    Square {
        IconBtn(if (h.on && !h.paused) Icons.pause else Icons.play, if (!h.on) "Start" else if (h.paused) "Resume" else "Pause",
            enabled = h.on || h.canStart) { if (h.on) h.pause(!h.paused) else h.requestStart() }
    }
    RateSquare(h.railCapacity)
}

@Composable
private fun Side(h: SendHost) {
    Spacer(Modifier.height(12.dp))
    val fill = LocalButtonsFill.current
    Row(if (fill) Modifier.fillMaxWidth() else Modifier, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        val grow = if (fill) Modifier.weight(1f) else Modifier
        if (!h.busy) Btn("Start", Kind.Primary, enabled = h.canStart, modifier = grow) { h.requestStart() }
        else {
            Btn("Stop", modifier = grow, pad = 16.dp) { h.stop() }
            if (h.pauseShown) {
                if (h.paused) Btn("Resume", Kind.Primary, pad = 16.dp) { h.pause(false) }
                else Btn("Pause", enabled = h.on, pad = 16.dp) { h.pause(true) }
            }
        }
        if (h.fullscreen) Btn("Fullscreen", pad = 16.dp) { h.toggleFull() }
    }
    val ln = h.line
    val color = if (ln.bad) Bad else Fg
    val named = ln.name
    // ai: from Start until Stop the state line and the figures keep a line each, empty or not (2026-11-02: Preparing,
    // ai: Sending, the first second's figures and Pause moved everything under them); idle, only what has words
    val held = h.busy
    if (named != null) NameLine(ln.before, named, ln.after, MaterialTheme.typography.titleMedium, color, Modifier.padding(top = 12.dp))
    else if (ln.before.isNotEmpty() || held) Text(ln.before, style = MaterialTheme.typography.titleMedium, color = color,
        maxLines = if (ln.bad) Int.MAX_VALUE else 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 12.dp))
    if (h.figures.isNotEmpty() || held) Text(h.figures, style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1, modifier = Modifier.padding(top = 2.dp))
    h.heat?.let { Warning(it, Modifier.padding(top = 8.dp)) }
    Spacer(Modifier.height(16.dp))
    Fold("Settings", h.folds.open(FoldKey.SendSettings), { h.folds.toggle(FoldKey.SendSettings) }) { Settings(h.settings) }
    if (h.experimental.isNotEmpty()) Fold("Experimental", h.folds.open(FoldKey.SendExperimental), { h.folds.toggle(FoldKey.SendExperimental) }) { Settings(h.experimental) }
    Fold("Developer Tools", h.folds.open(FoldKey.SendTools), { h.folds.toggle(FoldKey.SendTools) }) {
        Settings(h.tools)
        h.readout?.let { CodeBlock(it.text, it.size) }
    }
    AboutFold(h.about)
    Spacer(Modifier.height(24.dp))
}

// ai: The senders' file upload (2026-10-26, LocalSend's geometry; the web's ui.css .tile and .itile): the tiles (File,
// ai: Folder and where the platform has one Media) in one row stretched across the box in equal cells, and under them,
// ai: while the selection holds something, its card as wide: the count with Clear, then a tile an item (its icon, its
// ai: name with the stem cut and the ending kept, the line under it, an x that removes it), in rows stretched across the
// ai: card (TileRows, TILE dp the least), one row scrolling sideways in landscape; nothing changes while sending. At the
// ai: top of its box, 16 dp from its edges; only the card scrolls. The phone draws it in Compose; the desktop paints
// ai: the same with AWT on its heavyweight code canvas (nothing of Compose draws over that).
class UploadItem(val key: String, val name: String, val sub: String, val dir: Boolean)

@Composable
fun Upload(tiles: List<Act>, items: List<UploadItem>, enabled: Boolean, landscape: Boolean, onRemove: (String) -> Unit, onClear: () -> Unit) {
    BoxWithConstraints(Modifier.fillMaxSize().background(Bg)) {
        val w = maxWidth - 32.dp
        Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                tiles.forEach { t -> Tile(t.icon ?: Icons.file, t.label, (w - 12.dp * (tiles.size - 1)) / tiles.size, enabled = enabled, onClick = t.run) }
            }
            if (items.isNotEmpty()) Column(Modifier.width(w).weight(1f, fill = false).clip(RoundedCornerShape(12.dp)).background(Card)
                .padding(start = 12.dp, end = 12.dp, top = 4.dp, bottom = 12.dp)) {
                Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).padding(start = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("${items.size} ${if (items.size == 1) "item" else "items"}", style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
                    Btn("Clear", Kind.Text, enabled = enabled) { onClear() }
                }
                val item: @Composable (Int, Dp) -> Unit = { i, iw ->
                    val it = items[i]
                    ItemTile(it.name, it.sub, if (it.dir) Icons.folder else Icons.file, iw, enabled = enabled) { onRemove(it.key) }
                }
                if (landscape) {
                    // ai: one row at the columns' width (TileRows' rule), scrolling sideways past what fits
                    val gap = 8.dp; val inner = w - 24.dp
                    val cols = maxOf(1, ((inner + gap) / (TILE.dp + gap)).toInt())
                    Row(Modifier.padding(top = 4.dp).horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(gap)) {
                        items.indices.forEach { item(it, (inner - gap * (cols - 1)) / cols) }
                    }
                } else Box(Modifier.padding(top = 4.dp).weight(1f, fill = false).verticalScroll(rememberScrollState())) { TileRows(items.size, w - 24.dp, 8.dp, item) }
            }
        }
    }
}

// ai: the upload's height with one row of items: its padding, the tiles, the gap, the card's head and one row
val UPLOAD_ONE_ROW = (16 + 104 + 16 + 4 + 48 + 4 + 104 + 12 + 16).dp

const val TILE = 104

// ai: a Tile is a button TILE dp tall, an icon over one word; as wide as its row gives it
@Composable
fun Tile(icon: ImageVector, label: String, width: Dp, enabled: Boolean = true, onClick: () -> Unit) {
    val tint = if (enabled) Fg else Fg.copy(alpha = .38f)
    Column(Modifier.width(width).height(TILE.dp).clip(RoundedCornerShape(12.dp)).background(Soft).clickable(enabled = enabled, role = Role.Button, onClick = onClick),
        verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically), horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(icon, contentDescription = null, Modifier.size(32.dp), tint = tint)
        Text(label, style = MaterialTheme.typography.labelLarge, color = tint)
    }
}

// ai: an ItemTile is one item on the selection's card: its icon, its name in medium with the stem cut and the ending
// ai: kept (Readout.nameParts), the line under it regular, and an x at its corner that removes it
@Composable
fun ItemTile(name: String, sub: String, icon: ImageVector, width: Dp, enabled: Boolean = true, onRemove: () -> Unit) {
    val (stem, end) = Readout.nameParts(name)
    val name13 = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium)
    Box(Modifier.width(width).height(TILE.dp).clip(RoundedCornerShape(12.dp)).background(Bg)) {
        Column(Modifier.fillMaxSize().padding(8.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(icon, contentDescription = null, Modifier.padding(bottom = 6.dp).size(28.dp), tint = Muted)
            Row {
                Text(stem, style = name13, color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
                Text(end, style = name13, color = Fg, maxLines = 1)
            }
            Text(sub, style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Normal), color = Muted, maxLines = 1)
        }
        Box(Modifier.align(Alignment.TopEnd).padding(2.dp).size(28.dp).clip(CircleShape).clickable(enabled = enabled, role = Role.Button, onClick = onRemove)
            .semantics { contentDescription = "Remove $name" }, contentAlignment = Alignment.Center) {
            Icon(Icons.x, contentDescription = null, Modifier.size(16.dp), tint = if (enabled) Fg else Fg.copy(alpha = .38f))
        }
    }
}

// ai: n things in rows across `width`, each `gap` apart: as many columns as fit at TILE dp or more, each column stretched
// ai: to share the width, so a full row fills it and a short last row keeps the columns' width; cell(i, w) draws the i-th
// ai: at width w (the web's grid: repeat(auto-fill, minmax(104px, 1fr)))
@Composable
fun TileRows(n: Int, width: Dp, gap: Dp, cell: @Composable (Int, Dp) -> Unit) {
    val cols = maxOf(1, ((width + gap) / (TILE.dp + gap)).toInt())
    val w = (width - gap * (cols - 1)) / cols
    Column(verticalArrangement = Arrangement.spacedBy(gap)) {
        for (r in 0 until (n + cols - 1) / cols) Row(horizontalArrangement = Arrangement.spacedBy(gap)) {
            for (i in r * cols until minOf(n, (r + 1) * cols)) cell(i, w)
        }
    }
}
