package dev.lizard.receiver

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.material3.VerticalDivider
import androidx.compose.ui.unit.sp
import kotlin.math.floor

import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.border
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.SolidColor

// ai: The web's tokens and parts (lizard-web/ui.css; 2026-10-01): light, black and white, flat; one solid black button
// ai: on a screen, the others on a soft grey fill, minor ones text alone, every button a pill; a top bar with a back
// ai: arrow; lists of rows; cards on a soft fill. Colour only for an error and a finished transfer.
val Fg = Color(0xFF111111)
val Bg = Color(0xFFFFFFFF)
val Muted = Color(0xFF666666)
val Line = Color(0xFFE4E4E4)
val Soft = Color(0xFFF1F1F1)
val Card = Color(0xFFF6F6F6)
val Bad = Color(0xFFB00020)
val Good = Color(0xFF0A7A2F)

fun toneColour(t: Readout.Tone) = when (t) { Readout.Tone.Bad -> Bad; Readout.Tone.Good -> Good; else -> Fg }

// ai: The faces (2026-10-10): the web's, from liblizard/vendor/fonts (build.gradle.kts LizardFonts): Inter, one variable
// ai: file read at each weight the app sets by its axis, and JetBrains Mono for the readouts.
@OptIn(ExperimentalTextApi::class)
private fun inter(w: FontWeight) = Font(R.font.inter_variable, w, variationSettings = FontVariation.Settings(FontVariation.weight(w.weight)))
val Inter = FontFamily(listOf(FontWeight.Normal, FontWeight.Medium, FontWeight.SemiBold, FontWeight.Bold).map(::inter))
val Mono = FontFamily(Font(R.font.jetbrains_mono_regular))

// ai: The type: the web's sizes in sp, in Inter, and no letter spacing (Material's default runs 0.1 to 0.5 sp, which set
// ai: the app's text wider than the web's).
private fun TextStyle.flat() = copy(letterSpacing = 0.sp, fontFamily = Inter)
private val base = Typography()
val LizardType = Typography(
    displayLarge = base.displayLarge.flat(), displayMedium = base.displayMedium.flat(), displaySmall = base.displaySmall.flat(),
    headlineLarge = base.headlineLarge.flat(), headlineMedium = base.headlineMedium.flat(),
    headlineSmall = TextStyle(fontSize = 22.sp, lineHeight = 28.sp, fontWeight = FontWeight.SemiBold).flat(),
    titleLarge = TextStyle(fontSize = 20.sp, lineHeight = 28.sp, fontWeight = FontWeight.SemiBold).flat(),
    titleMedium = TextStyle(fontSize = 18.sp, lineHeight = 24.sp, fontWeight = FontWeight.SemiBold).flat(),
    titleSmall = TextStyle(fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.SemiBold).flat(),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp).flat(),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp).flat(),
    bodySmall = TextStyle(fontSize = 13.sp, lineHeight = 18.sp).flat(),
    labelLarge = TextStyle(fontSize = 15.sp, lineHeight = 20.sp, fontWeight = FontWeight.SemiBold).flat(),
    labelMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Medium).flat(),
    labelSmall = TextStyle(fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.Medium).flat(),
)
// ai: Material's surfaces all white (menus and dialogs float on a hairline, not a tint)
val LizardColours = lightColorScheme(primary = Fg, onPrimary = Bg, background = Bg, onBackground = Fg, surface = Bg, onSurface = Fg,
    surfaceVariant = Soft, onSurfaceVariant = Muted, outline = Line, outlineVariant = Line, error = Bad,
    surfaceContainerLowest = Bg, surfaceContainerLow = Bg, surfaceContainer = Bg, surfaceContainerHigh = Bg, surfaceContainerHighest = Bg)

// ai: The top bar every screen carries (the web's .bar): Home's mark and name (title null), or a back arrow and the
// ai: screen's title; the actions on the right (none since the gear went, 2026-10-02). The arrow and the last action sit
// ai: out in the gutter, so their ink lines up with the text below, as the web's do.
@Composable
fun TopBar(title: String?, onBack: (() -> Unit)?, actions: @Composable RowScope.() -> Unit = {}) {
    Column {
        Row(Modifier.fillMaxWidth().height(56.dp), verticalAlignment = Alignment.CenterVertically) {
            Row(Modifier.weight(1f).offset(x = if (onBack != null) (-12).dp else 0.dp), verticalAlignment = Alignment.CenterVertically) {
                if (onBack != null) IconBtn(R.drawable.ic_back, "Back", onClick = onBack)
                if (title == null) {
                    Image(painterResource(R.drawable.mark), contentDescription = null, Modifier.size(28.dp))
                    Spacer(Modifier.size(10.dp))
                    Text("LIZARD", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = Fg)
                } else Text(title, style = MaterialTheme.typography.titleLarge, color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.padding(start = if (onBack != null) 4.dp else 0.dp))
            }
            Row(Modifier.offset(x = 12.dp), verticalAlignment = Alignment.CenterVertically, content = actions)
        }
        HorizontalDivider(color = Line)
    }
}

// ai: An icon button: a 24 dp icon in a round 48 dp target, no border, the press a soft circle.
@Composable
fun IconBtn(icon: Int, label: String, tint: Color = Fg, enabled: Boolean = true, onClick: () -> Unit) {
    Box(Modifier.size(48.dp).clip(CircleShape).clickable(enabled = enabled, role = Role.Button, onClick = onClick).semantics { contentDescription = label },
        contentAlignment = Alignment.Center) {
        Icon(painterResource(icon), contentDescription = null, Modifier.size(24.dp), tint = if (enabled) tint else tint.copy(alpha = .38f))
    }
}

// ai: The sidebar collapser (2026-10-02, after the web's): an icon button whose chevrons point to where the panel goes;
// ai: `left` the panel's side (Receive's column; the sending screen's panel is on the right). Landscape only, as the web's.
@Composable
fun CollapseBtn(collapsed: Boolean, left: Boolean = true, onClick: () -> Unit) {
    Box(Modifier.size(48.dp).clip(CircleShape).clickable(role = Role.Button, onClick = onClick)
        .semantics { contentDescription = if (collapsed) "Show sidebar" else "Hide sidebar" }, contentAlignment = Alignment.Center) {
        Icon(painterResource(R.drawable.ic_collapse), contentDescription = null, Modifier.size(24.dp).rotate(if (collapsed == left) 180f else 0f), tint = Fg)
    }
}

// ai: The collapsed panel (the web's 56 px rail): the button at its top, a hairline under its row as the bar's, then
// ai: the caller's squares (Receive's: the camera, the rate, the file in; 2026-10-02), and `foot`, squares at the
// ai: bottom (Receive's Home, 2026-10-05, out of the bar so expanding the panel puts no back arrow under the finger).
@Composable
fun Rail(left: Boolean = true, onClick: () -> Unit, foot: @Composable () -> Unit = {}, squares: @Composable () -> Unit = {}) {
    Column(Modifier.width(56.dp).fillMaxHeight(), horizontalAlignment = Alignment.CenterHorizontally) {
        Box(Modifier.fillMaxWidth().height(56.dp), contentAlignment = Alignment.Center) { CollapseBtn(true, left, onClick) }
        HorizontalDivider(color = Line)
        squares()
        Spacer(Modifier.weight(1f))
        foot()
    }
}

@Composable
fun Square(content: @Composable () -> Unit) = Box(Modifier.size(56.dp), contentAlignment = Alignment.Center) { content() }

// ai: The landscape column's width (Receive and Send, 2026-10-05): 240 dp or a third of the screen's width where that is
// ai: less (320 and 0.45 until then, a fourth wider), or what its edge was dragged to (`chosen`, dp; 0 none), within
// ai: 180 dp (or `lo`, a screen's own floor: what its buttons need with no label folded, CellsFloor, 2026-10-06) and
// ai: three fifths of the screen.
fun SideWidth(screen: Dp, chosen: Float, lo: Dp = 180.dp): Dp {
    val low = maxOf(180.dp, lo)
    val hi = maxOf(low, screen * 0.6f)
    return (if (chosen > 0f) chosen.dp else minOf(240.dp, screen * 0.34f)).coerceIn(low, hi)
}

// ai: The column's floor (2026-10-06): what a screen's row of `cells` equal cells (weight 1f each) needs with no label
// ai: folded: `cells` times the widest label in the buttons' type plus their padding, the gaps between and the
// ai: column's own padding (a first form took each side's own widest, which left "Start camera" cut in its half). A
// ai: dragged column stops there, and the default is no narrower (SideWidth, SideEdge); every label counts whether or
// ai: not its button shows, so the column does not move when one appears.
@Composable
fun CellsFloor(labels: List<String>, cells: Int = 2): Dp {
    val tm = rememberTextMeasurer()
    val style = MaterialTheme.typography.labelLarge
    val density = LocalDensity.current
    val cell = labels.maxOf { with(density) { tm.measure(AnnotatedString(it), style).size.width.toDp() } } + 40.dp
    return cell * cells + 8.dp * (cells - 1) + 32.dp
}

// ai: The column's edge (2026-10-05): the hairline between the column and the camera or the code, in a strip wide enough
// ai: for a finger. Dragged sideways it sets the column's width: onDrag with the new width, from `width` at the drag's
// ai: start, no narrower than `lo` (the screen's floor); onDone on release (the caller keeps it).
@Composable
fun SideEdge(width: Dp, screen: Dp, lo: Dp = 180.dp, onStart: () -> Unit, onDrag: (Dp) -> Unit, onDone: () -> Unit) {
    val density = LocalDensity.current
    val w by rememberUpdatedState(width)
    val start by rememberUpdatedState(onStart)
    val drag by rememberUpdatedState(onDrag)
    val done by rememberUpdatedState(onDone)
    var at by remember { mutableStateOf(0.dp) }
    val low = maxOf(180.dp, lo)
    val state = rememberDraggableState { dx ->
        at += with(density) { dx.toDp() }
        drag(at.coerceIn(low, maxOf(low, screen * 0.6f)))
    }
    Box(Modifier.width(16.dp).fillMaxHeight().draggable(state, Orientation.Horizontal,
        onDragStarted = { at = w; start() }, onDragStopped = { done() }), contentAlignment = Alignment.Center) {
        VerticalDivider(color = Line)
    }
}

// ai: A rate in a rail's square: its figure over its unit (Readout.rate's no-break space the cut); empty for null
@Composable
fun RateSquare(kbs: Double?) = Square {
    if (kbs != null) {
        val p = Readout.rate(kbs).split('\u00a0')
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(p[0], style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, color = Fg)
            Text(p.getOrElse(1) { "" }, style = MaterialTheme.typography.labelSmall, color = Muted)
        }
    }
}

// ai: The share of a file in, in a rail's square (2026-10-09): its whole percent over "%", as the rate's figure over its
// ai: unit; the figures line's own percentage
@Composable
fun PctSquare(frac: Double) = Square {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text("${floor(100 * frac).toInt()}", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, color = Fg)
        Text("%", style = MaterialTheme.typography.labelSmall, color = Muted)
    }
}

// ai: The buttons (the web's button, .primary, .text, .danger): pills 48 dp tall. A label is one line and never
// ai: shortened (2026-10-06; two lines with an ellipsis before): a column is kept wide enough for it (CellsFloor).
enum class Kind { Primary, Tonal, Text, Danger }

@Composable
fun Btn(text: String, kind: Kind = Kind.Tonal, enabled: Boolean = true, modifier: Modifier = Modifier, onClick: () -> Unit) {
    val mod = modifier.heightIn(min = 48.dp)
    val label: @Composable () -> Unit = { Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 1, softWrap = false, overflow = TextOverflow.Visible) }
    when (kind) {
        Kind.Primary, Kind.Tonal -> Button(onClick, mod, enabled, CircleShape,
            ButtonDefaults.buttonColors(containerColor = if (kind == Kind.Primary) Fg else Soft, contentColor = if (kind == Kind.Primary) Bg else Fg,
                disabledContainerColor = Soft, disabledContentColor = Fg.copy(alpha = .38f)),
            elevation = null, contentPadding = PaddingValues(horizontal = 20.dp)) { label() }
        Kind.Text, Kind.Danger -> TextButton(onClick, mod, enabled, CircleShape,
            ButtonDefaults.textButtonColors(contentColor = if (kind == Kind.Danger) Bad else Fg, disabledContentColor = Fg.copy(alpha = .38f)),
            contentPadding = PaddingValues(horizontal = 12.dp)) { label() }
    }
}

@Composable
fun Meter(frac: Double?) {
    if (frac == null) return
    Box(Modifier.fillMaxWidth().padding(vertical = 8.dp).height(6.dp).clip(RoundedCornerShape(3.dp)).background(Soft)) {
        if (frac > 0) Box(Modifier.fillMaxWidth(frac.toFloat().coerceIn(0f, 1f)).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(Fg))
    }
}

// ai: A first-run tip (the web's .tip): a card on the soft fill, an info icon, its text, a text "Got it"; shown until one
// ai: is pressed (Home's Show tips again brings it back). Home's alone since 2026-10-02.
@Composable
fun TipCard(onGot: () -> Unit, content: @Composable ColumnScope.() -> Unit) {
    Row(Modifier.fillMaxWidth().padding(vertical = 8.dp).clip(RoundedCornerShape(12.dp)).background(Card)
        .padding(start = 16.dp, end = 16.dp, top = 14.dp, bottom = 2.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Icon(painterResource(R.drawable.ic_info), contentDescription = null, Modifier.size(24.dp), tint = Muted)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            content()
            Btn("Got it", Kind.Text, modifier = Modifier.offset(x = (-12).dp), onClick = onGot)
        }
    }
}

// ai: A row (the web's .item): a leading icon on a soft square, the title and the line under it, a trailing control;
// ai: the whole row its main action; a hairline under it.
@Composable
fun ListRow(title: String, sub: String? = null, lead: Int? = null, divider: Boolean = true, onClick: (() -> Unit)? = null,
            trailing: @Composable (() -> Unit)? = null) {
    Column {
        Row(Modifier.fillMaxWidth().heightIn(min = if (sub != null) 64.dp else 56.dp)
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier).padding(vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            if (lead != null) Box(Modifier.size(40.dp).clip(RoundedCornerShape(10.dp)).background(Soft), contentAlignment = Alignment.Center) {
                Icon(painterResource(lead), contentDescription = null, Modifier.size(24.dp), tint = Fg)
            }
            Column(Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = Fg, maxLines = 2, overflow = TextOverflow.Ellipsis)
                if (sub != null) Text(sub, style = MaterialTheme.typography.bodyMedium, color = Muted)
            }
            trailing?.invoke()
        }
        if (divider) HorizontalDivider(color = Line)
    }
}

// ai: A section's heading (the web's section h2): small, grey.
@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier) =
    Text(text, style = MaterialTheme.typography.titleSmall, color = Muted, modifier = modifier.padding(top = 24.dp, bottom = 4.dp))

// ai: The lab's text (the web's .lab and pre.readout): JetBrains Mono on the card fill.
@Composable
fun CodeBlock(text: String, size: Int = 13) =
    Text(text, fontFamily = Mono, fontSize = size.sp, lineHeight = (size + 5).sp, color = if (size < 13) Fg else Muted,
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp).clip(RoundedCornerShape(8.dp)).background(Card).padding(horizontal = 12.dp, vertical = 8.dp))

// ai: The heat warning (2026-10-01): a red triangle and the line "Thermal throttling. Speeds may slow down." Since
// ai: 2026-10-03 it shows when the phone has lowered the GPU's clock ceiling (Clocks.kt, under its top; the CPU's not)
// ai: and says so, "Thermal throttling. GPU at 646 of 1300 MHz."; where the GPU's clock cannot be read, from
// ai: Android's moderate thermal status (2) up with the words before. Since 2026-10-07, first of all, when the
// ai: camera's rate has fallen from its peak (Engine.cameraSlow): "Thermal throttling. Camera at 53 of 60 frames a
// ai: second.", the loss that costs the rate most. heatText is its line, null while nothing throttles.
fun heatText(status: Int, clocks: Clocks.Read?, camera: Engine.CameraSlow? = null): String? = when {
    camera != null -> "Thermal throttling. Camera at ${camera.now} of ${camera.max} frames a second."
    clocks != null -> if (clocks.throttled) Clocks.text(clocks) else null
    status >= android.os.PowerManager.THERMAL_STATUS_MODERATE -> "Thermal throttling. Speeds may slow down."
    else -> null
}

@Composable
fun HeatWarning(status: Int, clocks: Clocks.Read?, camera: Engine.CameraSlow? = null, modifier: Modifier = Modifier) {
    val text = heatText(status, clocks, camera) ?: return
    Row(modifier.semantics(mergeDescendants = true) {}, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Icon(painterResource(R.drawable.ic_alert), contentDescription = "Warning", Modifier.size(20.dp), tint = Bad)
        Text(text, style = MaterialTheme.typography.bodyMedium, color = Bad)
    }
}

// ai: The heat warning in a rail (2026-10-08): the red triangle alone in a square, a mark with no action, its line the
// ai: description.
@Composable
fun HeatSquare(text: String) = Square { Icon(painterResource(R.drawable.ic_alert), contentDescription = text, Modifier.size(24.dp), tint = Bad) }

// ai: A screen (2026-10-02): the bar fixed at the top,
// ai: what is under it scrolling on its own; one column at most 640 dp wide, centred.
@Composable
fun Page(title: String?, onBack: (() -> Unit)?, content: @Composable ColumnScope.() -> Unit) {
    Box(Modifier.fillMaxSize().background(Bg).safeDrawingPadding(), contentAlignment = Alignment.TopCenter) {
        Column(Modifier.widthIn(max = 640.dp).fillMaxWidth().fillMaxHeight().padding(horizontal = 16.dp)) {
            TopBar(title, onBack)
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
                content()
                Spacer(Modifier.height(24.dp))
            }
        }
    }
}

// ai: A row that opens (the web's details.dev, 2026-10-02: Settings and Developer Tools on Send and Receive): a hairline over
// ai: it, its title, a chevron that turns, and what it holds straight under it; its state the caller's, kept
// ai: (MainActivity.folds).
@Composable
fun Fold(title: String, open: Boolean, onToggle: () -> Unit, content: @Composable ColumnScope.() -> Unit) {
    Column(Modifier.fillMaxWidth()) {
        HorizontalDivider(color = Line)
        Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).clickable(role = Role.Button, onClick = onToggle), verticalAlignment = Alignment.CenterVertically) {
            Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = Fg, modifier = Modifier.weight(1f))
            Icon(painterResource(R.drawable.ic_chevron), contentDescription = null, Modifier.size(24.dp).rotate(if (open) 180f else 0f), tint = Muted)
        }
        if (open) {
            content()
            Spacer(Modifier.height(16.dp))
        }
    }
}

// ai: Settings' fields one rhythm (2026-10-02): 16 dp apart
// ai: (Fields), each its title over its control (14 sp, the web's label, 2026-10-06), the title row 36 dp whatever is
// ai: in it (the value on the right, `end` at its end), a group's heading (Group) 8 dp over its first
// ai: field.
@Composable
fun Fields(content: @Composable ColumnScope.() -> Unit) = Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(16.dp), content = content)

@Composable
fun Field(title: String, value: String = "", end: (@Composable () -> Unit)? = null, content: @Composable () -> Unit) {
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().heightIn(min = 36.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            // ai: the title on one line whatever the value's length; the value the rest of the row, cut at its end
            Text(title, style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1, softWrap = false)
            Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
                if (value.isNotEmpty()) Text(value, style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            end?.invoke()
        }
        Spacer(Modifier.height(4.dp))
        content()
    }
}

@Composable
fun Group(title: String, content: @Composable ColumnScope.() -> Unit) = Column(Modifier.fillMaxWidth()) {
    Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold, color = Fg, modifier = Modifier.padding(bottom = 8.dp))
    Fields(content)
}

// ai: A choice among a few in the web's shape (the pages' select, 2026-10-06; a row of chips until then): the title
// ai: over a 48 dp field on the soft fill, 8 dp corners, the chosen option's label and a chevron, a tap opening the
// ai: options as a menu. `blank` shows where no option is the value (the web's placeholder option); an option
// ai: `enabledFor` says no is greyed and inert.
@Composable
fun <T> Select(title: String, value: T, options: List<Pair<T, String>>, blank: String = "", enabled: Boolean = true,
               enabledFor: (T) -> Boolean = { true }, onChoose: (T) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth()) {
        Text(title, style = MaterialTheme.typography.bodyMedium, color = Muted)
        Spacer(Modifier.height(4.dp))
        Box(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).clip(RoundedCornerShape(8.dp)).background(Soft)
                .clickable(enabled = enabled, role = Role.Button) { open = true }.padding(start = 12.dp, end = 14.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Text(options.firstOrNull { it.first == value }?.second ?: blank, style = MaterialTheme.typography.bodyLarge,
                    color = if (enabled) Fg else Fg.copy(alpha = .38f), maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                Icon(painterResource(R.drawable.ic_chevron), contentDescription = null, Modifier.size(20.dp), tint = Muted)
            }
            DropdownMenu(open, { open = false }) {
                for ((k, label) in options) DropdownMenuItem({ Text(label, color = Fg) }, { open = false; if (k != value) onChoose(k) }, enabled = enabledFor(k))
            }
        }
    }
}

// ai: A text field in the web's input look (2026-10-06): the title over a 48 dp box on the soft fill, 8 dp corners, a
// ai: hairline in the text's colour while it has the focus, the placeholder grey while it is empty.
@Composable
fun TextInput(title: String, value: String, placeholder: String, onChange: (String) -> Unit) {
    var focused by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth()) {
        Text(title, style = MaterialTheme.typography.bodyMedium, color = Muted)
        Spacer(Modifier.height(4.dp))
        BasicTextField(value, onChange, Modifier.fillMaxWidth().onFocusChanged { focused = it.isFocused }, singleLine = true,
            textStyle = MaterialTheme.typography.bodyLarge.copy(color = Fg), cursorBrush = SolidColor(Fg)) { inner ->
            Box(Modifier.fillMaxWidth().heightIn(min = 48.dp).clip(RoundedCornerShape(8.dp)).background(Soft)
                .border(1.dp, if (focused) Fg else Color.Transparent, RoundedCornerShape(8.dp)).padding(horizontal = 12.dp), contentAlignment = Alignment.CenterStart) {
                if (value.isEmpty()) Text(placeholder, style = MaterialTheme.typography.bodyLarge, color = Muted)
                inner()
            }
        }
    }
}

// ai: The other tool, at the bar's end (the web's #swap, 2026-10-06): the swap arrows and its name, a text button.
@Composable
fun SwapBtn(label: String, onClick: () -> Unit) =
    TextButton(onClick, Modifier.heightIn(min = 48.dp), shape = CircleShape, colors = ButtonDefaults.textButtonColors(contentColor = Fg),
        contentPadding = PaddingValues(horizontal = 12.dp)) {
        Icon(painterResource(R.drawable.ic_swap), contentDescription = null, Modifier.size(24.dp), tint = Fg)
        Spacer(Modifier.width(8.dp))
        Text(label, style = MaterialTheme.typography.labelLarge, maxLines = 1, softWrap = false)
    }

// ai: A slider in the web's look (2026-10-02): a 6 dp track
// ai: with round ends, filled up to a round thumb that covers the end it stands on, as the Meter is drawn. Material's
// ai: own (1.3) cut the track around a bar thumb, so at either end the track's rounded cap was gone. The track slot
// ai: spans the thumb's centre from one end to the other (the Slider lays it out so), so the fill ends at the thumb.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Bar(value: Float, range: ClosedFloatingPointRange<Float>, steps: Int, choose: (Float) -> Unit) {
    Slider(value, choose, modifier = Modifier.fillMaxWidth(), steps = steps, valueRange = range,
        thumb = { Box(Modifier.size(20.dp).background(Fg, CircleShape)) },
        track = { st ->
            val span = st.valueRange.endInclusive - st.valueRange.start
            val f = if (span > 0) ((st.value - st.valueRange.start) / span).coerceIn(0f, 1f) else 0f
            Canvas(Modifier.fillMaxWidth().height(6.dp)) {
                val y = size.height / 2
                drawLine(Line, Offset(0f, y), Offset(size.width, y), strokeWidth = size.height, cap = StrokeCap.Round)
                drawLine(Fg, Offset(0f, y), Offset(size.width * f, y), strokeWidth = size.height, cap = StrokeCap.Round)
            }
        })
}
