package dev.lizard.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MenuDefaults
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.ui.text.font.FontWeight
import dev.lizard.readout.Readout
import kotlin.math.floor
import kotlinx.coroutines.delay

// ai: The parts of both apps (2026-10-31; each app's Parts.kt before, the Android app's of 2026-10-01 and the desktop's
// ai: port of it), the web's (lizard-web/ui.css): sizes the web's, a target LocalTarget's (48 dp under a finger, 40
// ai: under a mouse). Where the two apps drew a part differently one form serves both since 2026-10-31: the desktop's
// ai: (its weights, its chips, its field titles, its select, its code blocks' space) where the two differed in look.

// ai: The top bar (the web's .bar): Home's mark and LIZARD in bold (title null), or the screen's title, with a back arrow where
// ai: onBack is given; the actions on the right. The arrow and the last action sit out in the gutter, so their ink
// ai: lines up with the text below; a hairline under it.
@Composable
fun TopBar(title: String?, onBack: (() -> Unit)? = null, actions: @Composable RowScope.() -> Unit = {}) {
    val gutter = (LocalTarget.current - 24.dp) / 2
    Column {
        Row(Modifier.fillMaxWidth().height(56.dp), verticalAlignment = Alignment.CenterVertically) {
            Row(Modifier.weight(1f).offset(x = if (onBack != null) -gutter else 0.dp), verticalAlignment = Alignment.CenterVertically) {
                if (onBack != null) IconBtn(Icons.back, "Back", onClick = onBack)
                if (title == null) {
                    Image(Icons.mark, contentDescription = null, Modifier.size(28.dp))
                    Spacer(Modifier.size(10.dp))
                    Text("LIZARD", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = Fg)
                } else Text(title, style = MaterialTheme.typography.titleLarge, color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.padding(start = if (onBack != null) 4.dp else 0.dp))
            }
            Row(Modifier.offset(x = gutter), verticalAlignment = Alignment.CenterVertically, content = actions)
        }
        HorizontalDivider(color = Line)
    }
}

// ai: An icon button: a 24 dp icon in a round target, no border, the press a soft circle; `turn` degrees
@Composable
fun IconBtn(icon: ImageVector, label: String, enabled: Boolean = true, turn: Float = 0f, onClick: () -> Unit) {
    Box(Modifier.size(LocalTarget.current).clip(CircleShape).clickable(enabled = enabled, role = Role.Button, onClick = onClick).semantics { contentDescription = label },
        contentAlignment = Alignment.Center) {
        Icon(icon, contentDescription = null, Modifier.size(24.dp).rotate(turn), tint = if (enabled) Fg else Fg.copy(alpha = .38f))
    }
}

// ai: The sidebar collapser (the web's #collapse, 2026-10-02): chevrons pointing to where the column goes
@Composable
fun CollapseBtn(collapsed: Boolean, onClick: () -> Unit) =
    IconBtn(Icons.collapse, if (collapsed) "Show sidebar" else "Hide sidebar", turn = if (collapsed) 180f else 0f, onClick = onClick)

// ai: The collapsed column (the web's 56 px rail): the button that opens it at its top (`onClick`; on every platform,
// ai: the desktop's sash beside it), a hairline under its row as the bar's, then the caller's squares, and `foot` at its
// ai: bottom (Home: the only way home from a tool page in landscape, 2026-10-05).
val RailWidth = 56.dp

@Composable
fun Rail(onClick: () -> Unit, foot: @Composable () -> Unit = {}, squares: @Composable () -> Unit = {}) {
    Column(Modifier.width(RailWidth).fillMaxHeight(), horizontalAlignment = Alignment.CenterHorizontally) {
        Box(Modifier.fillMaxWidth().height(56.dp), contentAlignment = Alignment.Center) { CollapseBtn(true, onClick) }
        HorizontalDivider(color = Line)
        squares()
        Spacer(Modifier.weight(1f))
        foot()
    }
}

@Composable
fun Square(content: @Composable () -> Unit) = Box(Modifier.size(56.dp), contentAlignment = Alignment.Center) { content() }

// ai: A rate in a rail's square: its figure over its unit (Readout.rate's no-break space the cut); empty for null
@Composable
fun RateSquare(kbs: Double?) = Square {
    if (kbs != null) {
        val p = Readout.rate(kbs).split(' ')
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(p[0], style = MaterialTheme.typography.labelLarge, color = Fg)
            Text(p.getOrElse(1) { "" }, style = MaterialTheme.typography.labelSmall, color = Muted)
        }
    }
}

// ai: The share of a file in, in a rail's square (2026-10-09): its whole percent over "%", as the rate's figure over its
// ai: unit; the figures line's own percentage
@Composable
fun PctSquare(frac: Double) = Square {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text("${floor(100 * frac).toInt()}", style = MaterialTheme.typography.labelLarge, color = Fg)
        Text("%", style = MaterialTheme.typography.labelSmall, color = Muted)
    }
}

// ai: The other tool, at the bar's end (the web's #swap): the swap arrows and its name, a text button
@Composable
fun SwapBtn(label: String, onClick: () -> Unit) =
    TextButton(onClick, Modifier.heightIn(min = LocalTarget.current), shape = CircleShape, colors = ButtonDefaults.textButtonColors(contentColor = Fg),
        contentPadding = PaddingValues(horizontal = 12.dp)) {
        Icon(Icons.swap, contentDescription = null, Modifier.size(24.dp), tint = Fg)
        Spacer(Modifier.width(8.dp))
        Text(label, style = MaterialTheme.typography.labelLarge, maxLines = 1, softWrap = false)
    }

// ai: A file's progress (the web's .meter): a 6 dp track on the soft fill, round ends, filled in the foreground's colour
// ai: to frac; nothing for null, or its room alone where `hold` (a run's lines keep their place, 2026-11-02)
@Composable
fun Meter(frac: Double?, hold: Boolean = false) {
    if (frac == null) { if (hold) Spacer(Modifier.padding(vertical = 8.dp).height(6.dp)); return }
    Box(Modifier.fillMaxWidth().padding(vertical = 8.dp).height(6.dp).clip(RoundedCornerShape(3.dp)).background(Soft)) {
        if (frac > 0) Box(Modifier.fillMaxWidth(frac.toFloat().coerceIn(0f, 1f)).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(Fg))
    }
}

// ai: The web's spinner (recv.html #loading .spin): a 28 dp ring of 3 dp in the hairline's grey, its turning arc the
// ai: foreground's
@Composable
fun Spinner() = CircularProgressIndicator(Modifier.size(28.dp), color = Fg, strokeWidth = 3.dp, trackColor = Line)

// ai: A line that names a file (2026-10-23; the web's ui.mjs sayName): its head, up to the name and a comma after it,
// ai: on one line, the name's stem cut with an ellipsis where the line is too narrow and its ending kept whole
// ai: (Readout.nameParts); the rest after it on the same line, cut with an ellipsis where even the stem is gone (one
// ai: line always, 2026-11-02: the rest wrapping under it made the line two lines tall, then one, as its figures moved).
@Composable
fun NameLine(before: String, name: String, after: String, style: TextStyle, color: Color, modifier: Modifier = Modifier) {
    val (stem, end) = Readout.nameParts(name)
    val comma = after.startsWith(",")
    val rest = (if (comma) after.drop(1) else after).trim()
    // ai: the head, the ending and the rest measured first, in that order; the stem takes what is left
    Row(modifier) {
        if (before.isNotEmpty()) Text(before, style = style, color = color, maxLines = 1)
        Text(stem, style = style, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
        Text(end + if (comma) "," else "", style = style, color = color, maxLines = 1)
        if (rest.isNotEmpty()) Text(" $rest", style = style, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

// ai: A line that comes and goes with a poll, held still (2026-11-02): shown once it has held `on` ms, kept until it has
// ai: been gone `off` ms; a new text over one shown takes its place at once. The receivers' hints, whose rules sit on
// ai: a rate that wavers about the code's, came and went with each quarter-second poll (shown at once since: its room
// ai: is kept for the run once shown, Receive.kt, so only its words would blink).
@Composable
fun steady(text: String, on: Long = 1000, off: Long = 2000): String {
    var shown by remember { mutableStateOf("") }
    LaunchedEffect(text) {
        if (text.isNotEmpty()) { if (shown.isEmpty()) delay(on); shown = text }
        else { delay(off); shown = "" }
    }
    // ai: with no wait, a text in this frame, not the next (the effect's write lands a frame later)
    return if (on == 0L && text.isNotEmpty()) text else shown
}

// ai: A section's heading (the web's section h2): small, grey, 24 dp over it
@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier) =
    Text(text, style = MaterialTheme.typography.titleSmall, color = Muted, modifier = modifier.padding(top = 24.dp, bottom = 4.dp))

// ai: A row (the web's .item): a leading icon on a soft square, the title and the line under it, a trailing control;
// ai: the whole row its main action where it has one; a hairline under it where `divider` (the web's rows of a list).
// ai: The title one line in medium, cut with an ellipsis where the row is too narrow (2026-10-23: a long file name took two).
@Composable
fun ListRow(title: String, sub: String? = null, lead: ImageVector? = null, divider: Boolean = false, onClick: (() -> Unit)? = null,
            trailing: @Composable (() -> Unit)? = null) {
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().heightIn(min = if (sub != null) 64.dp else 56.dp)
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier).padding(vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            if (lead != null) Box(Modifier.size(40.dp).clip(RoundedCornerShape(10.dp)).background(Soft), contentAlignment = Alignment.Center) {
                Icon(lead, contentDescription = null, Modifier.size(24.dp), tint = Fg)
            }
            Column(Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis)
                if (sub != null) Text(sub, style = MaterialTheme.typography.bodyMedium, color = Muted)
            }
            trailing?.invoke()
        }
        if (divider) HorizontalDivider(color = Line)
    }
}

// ai: A row that opens (the web's details.dev, 2026-10-02: Settings, Experimental and Developer Tools on Send and
// ai: Receive): a hairline over it, its title in medium, a chevron that turns, and what it holds straight under it; its state the
// ai: caller's, kept by the app.
@Composable
fun Fold(title: String, open: Boolean, onToggle: () -> Unit, content: @Composable ColumnScope.() -> Unit) {
    Column(Modifier.fillMaxWidth()) {
        HorizontalDivider(color = Line)
        Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).clickable(role = Role.Button, onClick = onToggle), verticalAlignment = Alignment.CenterVertically) {
            Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = Fg, modifier = Modifier.weight(1f))
            Icon(Icons.chevron, contentDescription = null, Modifier.size(24.dp).rotate(if (open) 180f else 0f), tint = Muted)
        }
        if (open) {
            content()
            Spacer(Modifier.height(16.dp))
        }
    }
}

// ai: Settings' fields one rhythm (2026-10-02): 16 dp apart (Fields; a fold's whole content, its readouts and buttons
// ai: too, none spaced on its own).
@Composable
fun Fields(content: @Composable ColumnScope.() -> Unit) = Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(16.dp), content = content)

// ai: A slider in the web's look (2026-10-02): a 6 dp track with round ends, filled up to a round thumb that covers the
// ai: end it stands on, as the Meter is drawn (Material's own cut the track around a bar thumb). The track slot spans
// ai: the thumb's centre from one end to the other, so the fill ends at the thumb. choose: each value while dragged (the
// ai: label live); done: on release, where a change is committed (a re-pick or a configure at each step of a drag would
// ai: drop frames for nothing).
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Bar(value: Float, range: ClosedFloatingPointRange<Float>, steps: Int, done: (() -> Unit)? = null, choose: (Float) -> Unit) {
    Slider(value, choose, modifier = Modifier.fillMaxWidth(), steps = steps, valueRange = range, onValueChangeFinished = done,
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

// ai: A setting's field (2026-10-02): its title over its control (14 sp semibold, muted), the title row 36 dp whatever
// ai: is in it (the value on the right, cut at its end; the title on one line whatever the value's length), 4 dp over
// ai: the control, in the fields' rhythm (Fields)
@Composable
fun Field(title: String, value: String = "", content: @Composable () -> Unit) {
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().heightIn(min = 36.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = Muted, maxLines = 1, softWrap = false)
            Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
                if (value.isNotEmpty()) Text(value, style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
        Spacer(Modifier.height(4.dp))
        content()
    }
}

// ai: A choice among several in the web's shape (the pages' select, 2026-10-06): a field's title (Field: its 36 dp row)
// ai: over a field on the soft fill, 8 dp corners, the chosen option's label and an 18 dp chevron 12 dp from its end, a
// ai: press opening the options as the web's menu (.menu: as wide as the field, on white with a hairline round it, 12 dp
// ai: corners, no shadow, the chosen one on the soft fill). `blank` shows where no option is the value (the web's
// ai: placeholder option); an option `enabledFor` says no to is greyed and inert; `onOpen` runs as the menu opens (a
// ai: list read again then).
@Composable
fun <T> Select(title: String, value: T, options: List<Pair<T, String>>, blank: String = "", enabled: Boolean = true,
               enabledFor: (T) -> Boolean = { true }, onOpen: () -> Unit = {}, onChoose: (T) -> Unit) {
    var open by remember { mutableStateOf(false) }
    var width by remember { mutableStateOf(0) }
    val density = LocalDensity.current
    Field(title) {
        Box(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth().heightIn(min = LocalTarget.current).onSizeChanged { width = it.width }.clip(RoundedCornerShape(8.dp)).background(Soft)
                .clickable(enabled = enabled, role = Role.Button) { onOpen(); open = true }.padding(start = 12.dp, end = 12.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Text(options.firstOrNull { it.first == value }?.second ?: blank, style = MaterialTheme.typography.bodyLarge,
                    color = if (enabled) Fg else Fg.copy(alpha = .38f), maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                Icon(Icons.chevron, contentDescription = null, Modifier.size(18.dp), tint = Muted)
            }
            DropdownMenu(open, { open = false }, Modifier.width(with(density) { width.toDp() }).heightIn(max = 480.dp),
                shape = RoundedCornerShape(12.dp), containerColor = Bg, shadowElevation = 0.dp, border = BorderStroke(1.dp, Line)) {
                for ((k, label) in options) DropdownMenuItem({ Text(label, style = MaterialTheme.typography.bodyLarge, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                    { open = false; if (k != value) onChoose(k) }, if (k == value) Modifier.background(Soft) else Modifier, enabled = enabledFor(k),
                    colors = MenuDefaults.itemColors(textColor = Fg))
            }
        }
    }
}

// ai: A row's more menu (the web's .menu: the more icon, its actions in a list on white with a hairline round it, 12 dp
// ai: corners, no shadow, its right edge under the icon's): each action its label, whether it is a deletion's (red),
// ai: and what it does
class Act(val label: String, val danger: Boolean = false, val icon: ImageVector? = null, val run: () -> Unit)

@Composable
fun MoreMenu(label: String, acts: List<Act>) {
    var open by remember { mutableStateOf(false) }
    val target = LocalTarget.current
    Box {
        IconBtn(Icons.more, label) { open = true }
        DropdownMenu(open, { open = false }, Modifier.widthIn(min = 176.dp), offset = DpOffset(target - 176.dp, 4.dp), shape = RoundedCornerShape(12.dp),
            containerColor = Bg, shadowElevation = 0.dp, border = BorderStroke(1.dp, Line)) {
            for (a in acts) DropdownMenuItem({ Text(a.label, style = MaterialTheme.typography.bodyLarge) }, { open = false; a.run() },
                colors = MenuDefaults.itemColors(textColor = if (a.danger) Bad else Fg))
        }
    }
}

// ai: A question before a deletion (the web's .dialog): its title, Cancel and the red action; on white, 12 dp corners
@Composable
fun AskDialog(title: String, action: String, onDismiss: () -> Unit, onConfirm: () -> Unit) =
    AlertDialog(onDismissRequest = onDismiss,
        confirmButton = { Btn(action, Kind.Danger) { onDismiss(); onConfirm() } },
        dismissButton = { Btn("Cancel", Kind.Text, onClick = onDismiss) },
        title = { Text(title, style = MaterialTheme.typography.titleMedium, color = Fg) },
        shape = RoundedCornerShape(12.dp), containerColor = Bg)

// ai: The buttons (the web's button, .primary, .text, .danger): pills a target tall, the sides' room `pad`; a label one
// ai: line, cut with an ellipsis where the button is too narrow for it.
enum class Kind { Primary, Tonal, Text, Danger }

@Composable
fun Btn(text: String, kind: Kind = Kind.Tonal, enabled: Boolean = true, modifier: Modifier = Modifier, pad: Dp = 20.dp, onClick: () -> Unit) {
    val mod = modifier.heightIn(min = LocalTarget.current)
    val label: @Composable () -> Unit = { Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 1, overflow = TextOverflow.Ellipsis) }
    when (kind) {
        Kind.Primary, Kind.Tonal -> Button(onClick, mod, enabled, CircleShape,
            ButtonDefaults.buttonColors(containerColor = if (kind == Kind.Primary) Fg else Soft, contentColor = if (kind == Kind.Primary) Bg else Fg,
                disabledContainerColor = Soft, disabledContentColor = Fg.copy(alpha = .38f)),
            elevation = null, contentPadding = PaddingValues(horizontal = pad)) { label() }
        Kind.Text, Kind.Danger -> TextButton(onClick, mod, enabled, CircleShape,
            ButtonDefaults.textButtonColors(contentColor = if (kind == Kind.Danger) Bad else Fg, disabledContentColor = Fg.copy(alpha = .38f)),
            contentPadding = PaddingValues(horizontal = 12.dp)) { label() }
    }
}

// ai: The lab's text (the web's .lab and pre.readout): JetBrains Mono on the card fill, 6 dp of light above and below
// ai: it; at 13 sp the web's .lab line, muted, at 12 a readout in the foreground's colour.
@Composable
fun CodeBlock(text: String, size: Int = 13) =
    Text(text, fontFamily = Mono, fontSize = size.sp, lineHeight = (size + 5).sp, color = if (size < 13) Fg else Muted,
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp).clip(RoundedCornerShape(8.dp)).background(Card).padding(horizontal = 12.dp, vertical = 8.dp))

// ai: A choice among a few (Send's FPS, Ring, Codes, Encoder, VSync, Payload): 36 dp, 8 dp corners, on the soft fill, the
// ai: chosen one solid with its label semibold; Chips lays them in a row that wraps, 8 dp apart.
@Composable
fun Chip(text: String, on: Boolean, enabled: Boolean = true, onClick: () -> Unit) {
    Box(Modifier.heightIn(min = 36.dp).clip(RoundedCornerShape(8.dp)).background(if (on) Fg else Soft)
        .selectable(selected = on, enabled = enabled, role = Role.RadioButton, onClick = onClick).padding(horizontal = 14.dp),
        contentAlignment = Alignment.Center) {
        Text(text, style = MaterialTheme.typography.labelLarge, fontWeight = if (on) FontWeight.SemiBold else FontWeight.Normal,
            color = if (on) Bg else if (enabled) Fg else Fg.copy(alpha = .38f))
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun Chips(content: @Composable () -> Unit) =
    FlowRow(Modifier.selectableGroup(), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) { content() }

// ai: A line of text in the select's look (2026-10-27, the phone's address): the soft fill, 8 dp corners, the
// ai: placeholder muted while empty; done with the text at Enter, or as the focus leaves it changed
@Composable
fun TextInput(value: String, placeholder: String, done: (String) -> Unit) {
    var text by remember(value) { mutableStateOf(value) }
    val target = LocalTarget.current
    BasicTextField(text, { text = it }, singleLine = true, textStyle = MaterialTheme.typography.bodyLarge.copy(color = Fg), cursorBrush = SolidColor(Fg),
        modifier = Modifier.fillMaxWidth().heightIn(min = target).clip(RoundedCornerShape(8.dp)).background(Soft)
            .onPreviewKeyEvent { e -> if (e.key == Key.Enter && e.type == KeyEventType.KeyUp) { done(text.trim()); true } else false }
            .onFocusChanged { if (!it.isFocused && text.trim() != value) done(text.trim()) },
        decorationBox = { inner ->
            Box(Modifier.padding(horizontal = 12.dp).heightIn(min = target), contentAlignment = Alignment.CenterStart) {
                if (text.isEmpty()) Text(placeholder, style = MaterialTheme.typography.bodyLarge, color = Muted)
                inner()
            }
        })
}
