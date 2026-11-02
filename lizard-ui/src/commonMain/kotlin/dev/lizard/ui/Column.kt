package dev.lizard.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.hoverable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.BoxWithConstraintsScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay

// ai: The tool screens' column (Receive and Send, the web's landscape pages): its width as dragged, kept by the app.
// ai: `default` the app's own when none is dragged (the desktop's the web's 20rem, 320 dp; the phone's 240 dp or a
// ai: third of the screen where that is less), `dp` what its edge was dragged to (0 none), `save` keeps it. `stretch`:
// ai: the column's rows of buttons fill its width (the phone's), or each button takes its label's width (the desktop's,
// ai: 2026-11-01: LocalButtonsFill). The bar, the folds and the rest stretch either way.
interface ColumnWidth {
    var dp: Float
    var dragging: Boolean
    val stretch: Boolean get() = true
    // ai: The column's edge as an editor's sash, the desktop's (2026-11-01): dragged it resizes the column, pulled under half
    // ai: its floor it closes to the rail, and pulled out from the rail it opens again; no collapse button. False (the
    // ai: phone's): the collapse button alone and no edge, a drag at the screen's side being the system's swipe.
    val sash: Boolean get() = false
    fun default(screen: Dp): Dp
    fun save()
}

// ai: The landscape column's width (2026-10-05): its default, or what its edge was dragged to, within 180 dp (or `lo`,
// ai: a screen's own floor: what its widest row of buttons needs, RowFloor) and three fifths of the screen.
fun SideWidth(screen: Dp, chosen: Float, default: Dp, lo: Dp = 180.dp): Dp {
    val low = maxOf(180.dp, lo)
    val hi = maxOf(low, screen * 0.6f)
    return (if (chosen > 0f) chosen.dp else default).coerceIn(low, hi)
}

// ai: The dragged column's floor: what a screen's widest row of buttons needs with no label cut, each button (the
// ai: labels it may show, the room either side of them) its widest label in the buttons' type with that room, no
// ai: narrower than a button's least, 8 dp between them and the column's padding; every label counts whether or not
// ai: it shows, so the floor does not move when one appears. The edge stops there, so a button that fills the rest of
// ai: its row (Stop, Open) is never crushed by its neighbours.
@Composable
fun RowFloor(buttons: List<Pair<List<String>, Dp>>): Dp {
    val tm = rememberTextMeasurer()
    val style = MaterialTheme.typography.labelLarge
    val density = LocalDensity.current
    fun label(s: String) = with(density) { tm.measure(AnnotatedString(s), style).size.width.toDp() }
    return buttons.fold(0.dp) { a, (labels, pad) -> a + maxOf(ButtonDefaults.MinWidth, labels.maxOf(::label) + pad * 2) } +
        8.dp * (buttons.size - 1) + SidePad * 2
}

// ai: The column's light at its start and before its hairline (its bar and its scrolling part alike)
val SidePad = 12.dp

// ai: Whether a row of buttons in the column fills its width (ColumnWidth.stretch) or each button is its label's width
val LocalButtonsFill = staticCompositionLocalOf { true }

// ai: The sash's margin past its hairline, the column's own and drawn in nothing: the room its wide line (4 dp, the hairline
// ai: and 3 more) grows into, so a view in the main area that Compose cannot draw over hides none of it (2026-11-02)
val EdgeRoom = 3.dp

// ai: The edge's strip: a finger's width past the column's hairline, over its margin (EdgeRoom) and the main area's
// ai: side, taking no room of its own
val EdgeStrip = 20.dp

// ai: The column's edge as a view the main area holds takes it, where Compose can neither draw nor hear over that view
// ai: (the desktop's code canvas, a heavyweight): the pointer within EdgeStrip less EdgeRoom of the view's left edge, the
// ai: strip's part over it, is over the edge (`hover`), and a press there dragged sideways drags it (`start`, `drag` by the pointer's move, `done` at the
// ai: release). Given to the main area where the edge is a sash (LocalEdgeGrip), open or collapsed; null otherwise.
interface EdgeGrip {
    fun hover(on: Boolean)
    fun start()
    fun drag(dx: Dp)
    fun done()
}

val LocalEdgeGrip = staticCompositionLocalOf<EdgeGrip?> { null }

// ai: The edge's drag and hover, one for the strip and for a view that takes it (EdgeGrip): a drag from the width at
// ai: its start, no narrower than `low` (the screen's floor) nor wider than three fifths of the screen. As a sash
// ai: (ColumnWidth.sash) the drag under half of `low` closes the column to the rail and back over it opens it again,
// ai: `onSnap` the column's collapse toggled at each crossing (`shut` follows it within the drag, before the screen
// ai: has composed the new state); the width is left as it was while closed.
private class Edge : EdgeGrip {
    var width = 0.dp
    var low = 180.dp
    var hi = 180.dp
    var sash = false
    var collapsed = false
    var onStart: () -> Unit = {}
    var onDrag: (Dp) -> Unit = {}
    var onDone: () -> Unit = {}
    var onSnap: () -> Unit = {}
    private var at = 0.dp
    private var shut = false
    var held by mutableStateOf(false)
    var near by mutableStateOf(false)
    var hoverLit by mutableStateOf(false)   // ai: under the pointer for 300 ms
    override fun hover(on: Boolean) { near = on }
    override fun start() { at = width; shut = collapsed; held = true; onStart() }
    override fun drag(dx: Dp) {
        at += dx
        val close = sash && at < low / 2
        if (close != shut) { shut = close; onSnap() }
        if (!close) onDrag(at.coerceIn(low, hi))
    }
    override fun done() { held = false; onDone() }
}

@Composable
private fun rememberEdge(width: Dp, screen: Dp, lo: Dp, sash: Boolean, collapsed: Boolean, onSnap: () -> Unit, onStart: () -> Unit,
                         onDrag: (Dp) -> Unit, onDone: () -> Unit): Edge {
    val e = remember { Edge() }
    e.width = width
    e.low = maxOf(180.dp, lo)
    e.hi = maxOf(e.low, screen * 0.6f)
    e.sash = sash; e.collapsed = collapsed
    e.onSnap = onSnap; e.onStart = onStart; e.onDrag = onDrag; e.onDone = onDone
    return e
}

// ai: The column's edge (2026-10-05): the column's hairline (drawn in the row, ToolLandscape), and over the main area
// ai: beside it an invisible strip wide enough for a finger, laid over what is there. Dragged sideways it sets the
// ai: column's width; onDone on release (the app keeps it; Send re-picks). Under the pointer for 300 ms (the strip's
// ai: or a view's that takes it, EdgeGrip), and while dragged, the hairline grows to a line 4 dp wide in a pressed
// ai: button's grey (Pressed), widening into the column's own margin past its hairline (EdgeRoom, the main area from its
// ai: far side; 2026-11-02: over the column's side before, so that the desktop's heavyweight code canvas, which nothing
// ai: of Compose draws over, hid none of it), as an editor's sash. The pointer over the strip the app's left-right
// ai: resize cursor (LocalResizeIcon). `modifier` places it at the hairline's left side.
@Composable
private fun SideEdge(e: Edge, modifier: Modifier) {
    val density = LocalDensity.current
    val state = rememberDraggableState { dx -> e.drag(with(density) { dx.toDp() }) }
    val hover = remember { MutableInteractionSource() }
    val hovered by hover.collectIsHoveredAsState()
    val over = hovered || e.near
    LaunchedEffect(over) { if (over) { delay(300); e.hoverLit = true } else e.hoverLit = false }
    val on = e.hoverLit || e.held
    val thick by animateDpAsState(if (on) 4.dp else 1.dp, tween(120))
    val colour by animateColorAsState(if (on) Pressed else Line, tween(120))
    Box(modifier.fillMaxHeight()) {
        Box(Modifier.width(thick).fillMaxHeight().background(colour))
        Box(Modifier.padding(start = 1.dp).width(EdgeStrip).fillMaxHeight().hoverable(hover).pointerHoverIcon(LocalResizeIcon.current)
            .draggable(state, Orientation.Horizontal, onDragStarted = { e.start() }, onDragStopped = { e.done() }))
    }
}

// ai: A tool screen in landscape (Receive and Send, the web's pages; 2026-10-02): the column on the left (the bar over the
// ai: rest, which scrolls on its own, the platform's scrollbar at its edge), its hairline, and the main area on the right
// ai: from the hairline (as a sash from EdgeRoom past it), `mainPadding` inside it; collapsed, the column is the 56 dp rail (its squares, Home at its foot)
// ai: beside the hairline. Its edge a sash on the desktop (ColumnWidth.sash: no collapse button in the column's bar, the
// ai: edge at the rail's hairline too), the collapse button alone on the phone; the rail's button opens it on both.
// ai: `bare`: the main area alone, the column, rail, hairline, padding and edge gone, the main area's content where it was
// ai: in the tree (the desktop's full screen, 2026-11-02: SendHost.codeAloneInPlace).
@Composable
fun ToolLandscape(title: String, collapsed: Boolean, onCollapse: () -> Unit, onHome: () -> Unit, swap: (@Composable () -> Unit)?,
                  width: ColumnWidth, floor: Dp, mainPadding: Dp = 16.dp, onResized: () -> Unit = {}, bare: Boolean = false,
                  rail: @Composable () -> Unit, side: @Composable () -> Unit, main: @Composable BoxWithConstraintsScope.() -> Unit) {
    BoxWithConstraints(Modifier.fillMaxSize().background(Bg).safeDrawingPadding()) {
        val screen = maxWidth
        val sideW = SideWidth(screen, width.dp, width.default(screen), floor)
        val sash = width.sash
        val shown = if (collapsed) RailWidth else sideW
        val edge = rememberEdge(shown, screen, floor, sash, collapsed, onSnap = onCollapse, onStart = { width.dragging = true },
            onDrag = { width.dp = it.value }, onDone = { width.dragging = false; width.save(); onResized() })
        Row(Modifier.fillMaxSize()) {
            if (!bare) {
                if (collapsed) Rail(onClick = onCollapse, foot = { Square { IconBtn(Icons.back, "Home", onClick = onHome) } }) { rail() }
                else Column(Modifier.width(sideW).fillMaxHeight()) {
                    Box(Modifier.padding(horizontal = SidePad)) { TopBar(title) { swap?.invoke(); if (!sash) CollapseBtn(false, onCollapse) } }
                    Box(Modifier.weight(1f).fillMaxWidth()) {
                        val scroll = rememberScrollState()
                        Column(Modifier.fillMaxSize().verticalScroll(scroll).padding(horizontal = SidePad)) {
                            CompositionLocalProvider(LocalButtonsFill provides width.stretch) { side() }
                        }
                        LocalScrollbar.current(scroll, Modifier.align(Alignment.CenterEnd).fillMaxHeight())
                    }
                }
                VerticalDivider(color = Line)
                // ai: as a sash, the room the edge's wide line grows into: the column's, invisible, the main area past it
                if (sash) Spacer(Modifier.width(EdgeRoom))
            }
            BoxWithConstraints(Modifier.weight(1f).fillMaxHeight().padding(if (bare) 0.dp else mainPadding), contentAlignment = Alignment.Center) {
                CompositionLocalProvider(LocalEdgeGrip provides if (sash && !bare) edge else null) { main() }
            }
        }
        if (sash && !bare) SideEdge(edge, Modifier.offset(x = shown))
    }
}

// ai: A tool screen upright (the phone's portrait): the bar with Home's arrow and the swap, then one column that scrolls,
// ai: the main area at its top
@Composable
fun ToolPortrait(title: String, onHome: () -> Unit, swap: @Composable RowScope.() -> Unit, content: @Composable () -> Unit) {
    Column(Modifier.fillMaxSize().background(Bg).safeDrawingPadding().padding(horizontal = 16.dp)) {
        TopBar(title, onBack = onHome, actions = swap)
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            Spacer(Modifier.height(8.dp))
            content()
        }
    }
}
