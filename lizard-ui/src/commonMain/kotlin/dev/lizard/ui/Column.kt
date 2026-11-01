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
// ai: the column's rows of buttons stretch with a column dragged wider than its default (the phone's), or keep the
// ai: default's width from its start (the desktop's, 2026-11-01: LocalButtonsWidest); narrower, they shrink with it. The
// ai: bar, the folds and the rest stretch either way.
interface ColumnWidth {
    var dp: Float
    var dragging: Boolean
    val stretch: Boolean get() = true
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

// ai: The widest a row of buttons in the column is drawn (ColumnWidth.stretch): unspecified, the column's width
val LocalButtonsWidest = staticCompositionLocalOf { Dp.Unspecified }

// ai: The edge's strip: a finger's width over the main area's side of the column's hairline, taking no room of its own
// ai: (the main area starts at the hairline, its own padding after it, as before the column was dragged)
val EdgeStrip = 20.dp

// ai: The column's edge as a view the main area holds takes it, where Compose can neither draw nor hear over that view
// ai: (the desktop's code canvas, a heavyweight): the pointer within EdgeStrip of the view's left edge is over the
// ai: edge (`hover`), and a press there dragged sideways drags it (`start`, `drag` by the pointer's move, `done` at the
// ai: release). Given to the main area while the column is open (LocalEdgeGrip), null otherwise.
interface EdgeGrip {
    fun hover(on: Boolean)
    fun start()
    fun drag(dx: Dp)
    fun done()
}

val LocalEdgeGrip = staticCompositionLocalOf<EdgeGrip?> { null }

// ai: The edge's drag and hover, one for the strip and for a view that takes it (EdgeGrip): a drag from the width at
// ai: its start, no narrower than `low` (the screen's floor) nor wider than three fifths of the screen
private class Edge : EdgeGrip {
    var width = 0.dp
    var low = 180.dp
    var hi = 180.dp
    var onStart: () -> Unit = {}
    var onDrag: (Dp) -> Unit = {}
    var onDone: () -> Unit = {}
    private var at = 0.dp
    var held by mutableStateOf(false)
    var near by mutableStateOf(false)
    override fun hover(on: Boolean) { near = on }
    override fun start() { at = width; held = true; onStart() }
    override fun drag(dx: Dp) { at += dx; onDrag(at.coerceIn(low, hi)) }
    override fun done() { held = false; onDone() }
}

@Composable
private fun rememberEdge(width: Dp, screen: Dp, lo: Dp, onStart: () -> Unit, onDrag: (Dp) -> Unit, onDone: () -> Unit): Edge {
    val e = remember { Edge() }
    e.width = width
    e.low = maxOf(180.dp, lo)
    e.hi = maxOf(e.low, screen * 0.6f)
    e.onStart = onStart; e.onDrag = onDrag; e.onDone = onDone
    return e
}

// ai: The column's edge (2026-10-05): the column's hairline (drawn in the row, ToolLandscape), and over the main area
// ai: beside it an invisible strip wide enough for a finger, laid over what is there. Dragged sideways it sets the
// ai: column's width; onDone on release (the app keeps it; Send re-picks). Under the pointer for 300 ms (the strip's
// ai: or a view's that takes it, EdgeGrip), and while dragged, the hairline grows to a line 4 dp wide in a pressed
// ai: button's grey (Pressed), widening over the column's side so a heavyweight view beside it hides none of it, as an
// ai: editor's sash. The pointer over the strip the app's left-right resize cursor (LocalResizeIcon). `modifier` places
// ai: it with the hairline's right side 4 dp from its start.
@Composable
private fun SideEdge(e: Edge, modifier: Modifier) {
    val density = LocalDensity.current
    val state = rememberDraggableState { dx -> e.drag(with(density) { dx.toDp() }) }
    val hover = remember { MutableInteractionSource() }
    val hovered by hover.collectIsHoveredAsState()
    var lit by remember { mutableStateOf(false) }
    val over = hovered || e.near
    LaunchedEffect(over) { if (over) { delay(300); lit = true } else lit = false }
    val on = lit || e.held
    val thick by animateDpAsState(if (on) 4.dp else 1.dp, tween(120))
    val colour by animateColorAsState(if (on) Pressed else Line, tween(120))
    Row(modifier.fillMaxHeight()) {
        Box(Modifier.width(4.dp).fillMaxHeight(), contentAlignment = Alignment.CenterEnd) { Box(Modifier.width(thick).fillMaxHeight().background(colour)) }
        Box(Modifier.width(EdgeStrip).fillMaxHeight().hoverable(hover).pointerHoverIcon(LocalResizeIcon.current).draggable(state, Orientation.Horizontal,
            onDragStarted = { e.start() }, onDragStopped = { e.done() }))
    }
}

// ai: A tool screen in landscape (Receive and Send, the web's pages; 2026-10-02): the column on the left (the bar over the
// ai: rest, which scrolls on its own, the platform's scrollbar at its edge), its hairline, its edge dragged to resize it,
// ai: and the main area on the right from the hairline, `mainPadding` inside it; collapsed, the column is the 56 dp rail
// ai: (its squares, Home at its foot) beside the hairline.
@Composable
fun ToolLandscape(title: String, collapsed: Boolean, onCollapse: () -> Unit, onHome: () -> Unit, swap: (@Composable () -> Unit)?,
                  width: ColumnWidth, floor: Dp, mainPadding: Dp = 16.dp, onResized: () -> Unit = {}, rail: @Composable () -> Unit,
                  side: @Composable () -> Unit, main: @Composable BoxWithConstraintsScope.() -> Unit) {
    BoxWithConstraints(Modifier.fillMaxSize().background(Bg).safeDrawingPadding()) {
        val screen = maxWidth
        val sideW = SideWidth(screen, width.dp, width.default(screen), floor)
        val buttons = if (width.stretch) Dp.Unspecified else width.default(screen) - SidePad * 2
        val edge = rememberEdge(sideW, screen, floor, onStart = { width.dragging = true }, onDrag = { width.dp = it.value },
            onDone = { width.dragging = false; width.save(); onResized() })
        Row(Modifier.fillMaxSize()) {
            if (collapsed) Rail(onClick = onCollapse, foot = { Square { IconBtn(Icons.back, "Home", onClick = onHome) } }) { rail() }
            else Column(Modifier.width(sideW).fillMaxHeight()) {
                Box(Modifier.padding(horizontal = SidePad)) { TopBar(title) { swap?.invoke(); CollapseBtn(false, onCollapse) } }
                Box(Modifier.weight(1f).fillMaxWidth()) {
                    val scroll = rememberScrollState()
                    Column(Modifier.fillMaxSize().verticalScroll(scroll).padding(horizontal = SidePad)) {
                        CompositionLocalProvider(LocalButtonsWidest provides buttons) { side() }
                    }
                    LocalScrollbar.current(scroll, Modifier.align(Alignment.CenterEnd).fillMaxHeight())
                }
            }
            VerticalDivider(color = Line)
            BoxWithConstraints(Modifier.weight(1f).fillMaxHeight().padding(mainPadding), contentAlignment = Alignment.Center) {
                CompositionLocalProvider(LocalEdgeGrip provides if (collapsed) null else edge) { main() }
            }
        }
        if (!collapsed) SideEdge(edge, Modifier.offset(x = sideW - 3.dp))
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
