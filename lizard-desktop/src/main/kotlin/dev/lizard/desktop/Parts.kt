package dev.lizard.desktop

import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
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
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ai: The look of the web's pages and the Android app (lizard-web/ui.css; lizard-android/.../Parts.kt, ported without Android
// ai: APIs; 2026-10-01): light, black and white, flat; one solid black button on a screen, the others on a soft grey
// ai: fill, minor ones text alone, every button a pill;
// ai: a top bar; cards on a soft fill. Colour only for an error. The sizes are the web's under a mouse (ui.css
// ai: pointer:fine): buttons and icon targets 40 dp, where the phone's are 48.
val Fg = Color(0xFF111111)
val Bg = Color(0xFFFFFFFF)
val Muted = Color(0xFF666666)
val Line = Color(0xFFE4E4E4)
val Soft = Color(0xFFF1F1F1)
val Card = Color(0xFFF6F6F6)
val Bad = Color(0xFFB00020)

// ai: The type: the web's sizes in sp, and no letter spacing (Material's default runs 0.1 to 0.5 sp).
private fun TextStyle.flat() = copy(letterSpacing = 0.sp)
private val base = Typography()
val LizardType = Typography(
    displayLarge = base.displayLarge.flat(), displayMedium = base.displayMedium.flat(), displaySmall = base.displaySmall.flat(),
    headlineLarge = base.headlineLarge.flat(), headlineMedium = base.headlineMedium.flat(),
    headlineSmall = TextStyle(fontSize = 22.sp, lineHeight = 28.sp, fontWeight = FontWeight.SemiBold),
    titleLarge = TextStyle(fontSize = 20.sp, lineHeight = 28.sp, fontWeight = FontWeight.SemiBold),
    titleMedium = TextStyle(fontSize = 18.sp, lineHeight = 24.sp, fontWeight = FontWeight.SemiBold),
    titleSmall = TextStyle(fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.SemiBold),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    bodySmall = TextStyle(fontSize = 13.sp, lineHeight = 18.sp),
    labelLarge = TextStyle(fontSize = 15.sp, lineHeight = 20.sp, fontWeight = FontWeight.SemiBold),
    labelMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Medium),
    labelSmall = TextStyle(fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.Medium),
)
// ai: Material's surfaces all white (menus float on a hairline, not a tint)
val LizardColours = lightColorScheme(primary = Fg, onPrimary = Bg, background = Bg, onBackground = Fg, surface = Bg, onSurface = Fg,
    surfaceVariant = Soft, onSurfaceVariant = Muted, outline = Line, outlineVariant = Line, error = Bad,
    surfaceContainerLowest = Bg, surfaceContainerLow = Bg, surfaceContainer = Bg, surfaceContainerHigh = Bg, surfaceContainerHighest = Bg)

// ai: The icons: Feather's (MIT) on a 24 grid, 2 px strokes, as the web pages and the app draw them; and Lizard's mark
// ai: (lizard-web/icon.svg: the corner marks, the dark ring between them, a grey picture, on white), the window's icon.
object Icons {
    private fun stroked(name: String, vararg d: String) = ImageVector.Builder(name, 24.dp, 24.dp, 24f, 24f).apply {
        for (p in d) addPath(addPathNodes(p), stroke = SolidColor(Fg), strokeLineWidth = 2f, strokeLineCap = StrokeCap.Round, strokeLineJoin = StrokeJoin.Round)
    }.build()
    val chevron = stroked("chevron", "M6,9l6,6 6,-6")
    val collapse = stroked("collapse", "M11,17l-5,-5 5,-5M18,17l-5,-5 5,-5")
    val play = stroked("play", "M6,3l14,9 -14,9z")
    val pause = stroked("pause", "M6,4h4v16h-4z M14,4h4v16h-4z")
    val file = stroked("file", "M14,2H6a2,2 0,0 0,-2 2v16a2,2 0,0 0,2 2h12a2,2 0,0 0,2 -2V8z", "M14,2v6h6")
    val mark = ImageVector.Builder("mark", 64.dp, 64.dp, 64f, 64f).apply {
        addPath(addPathNodes("M0 0h64v64h-64z"), fill = SolidColor(Color.White))
        addPath(addPathNodes("M14 14h9v9h-9zM41 14h9v9h-9zM14 41h9v9h-9zM41 41h9v9h-9zM25 16h14v3H25zM25 45h14v3H25zM16 25h3v14h-3zM45 25h3v14h-3z"), fill = SolidColor(Fg))
        addPath(addPathNodes("M25 25h7v7h-7zM32 32h7v7h-7z"), fill = SolidColor(Color(0xFF777777)))
        addPath(addPathNodes("M32 25h7v7h-7zM25 32h7v7h-7z"), fill = SolidColor(Color(0xFFBBBBBB)))
    }.build()
}

// ai: The top bar (the web's .bar): the screen's title, the actions on the right, the last one out in the gutter so its
// ai: ink lines up with the text below; a hairline under it.
@Composable
fun TopBar(title: String, actions: @Composable RowScope.() -> Unit = {}) {
    Column {
        Row(Modifier.fillMaxWidth().height(56.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(title, style = MaterialTheme.typography.titleLarge, color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
            Row(Modifier.offset(x = 8.dp), verticalAlignment = Alignment.CenterVertically, content = actions)
        }
        HorizontalDivider(color = Line)
    }
}

// ai: An icon button: a 24 dp icon in a round 40 dp target, no border, the press a soft circle.
@Composable
fun IconBtn(icon: ImageVector, label: String, tint: Color = Fg, enabled: Boolean = true, turn: Float = 0f, onClick: () -> Unit) {
    Box(Modifier.size(40.dp).clip(CircleShape).clickable(enabled = enabled, role = Role.Button, onClick = onClick).semantics { contentDescription = label },
        contentAlignment = Alignment.Center) {
        Icon(icon, contentDescription = null, Modifier.size(24.dp).rotate(turn), tint = if (enabled) tint else tint.copy(alpha = .38f))
    }
}

// ai: The sidebar collapser (the web's #collapse, 2026-10-02): chevrons pointing to where the column goes.
@Composable
fun CollapseBtn(collapsed: Boolean, onClick: () -> Unit) =
    IconBtn(Icons.collapse, if (collapsed) "Show sidebar" else "Hide sidebar", turn = if (collapsed) 180f else 0f, onClick = onClick)

// ai: The collapsed column (the web's 56 px rail): the collapser at its top, a hairline under its row as the bar's, then
// ai: the caller's squares.
@Composable
fun Rail(onClick: () -> Unit, squares: @Composable () -> Unit = {}) {
    Column(Modifier.width(56.dp).fillMaxHeight(), horizontalAlignment = Alignment.CenterHorizontally) {
        Box(Modifier.fillMaxWidth().height(56.dp), contentAlignment = Alignment.Center) { CollapseBtn(true, onClick) }
        HorizontalDivider(color = Line)
        squares()
    }
}

@Composable
fun Square(content: @Composable () -> Unit) = Box(Modifier.size(56.dp), contentAlignment = Alignment.Center) { content() }

// ai: A rate in a rail's square: its figure over its unit (Fmt.rate's no-break space the cut); empty for null
@Composable
fun RateSquare(kbs: Double?) = Square {
    if (kbs != null) {
        val p = Fmt.rate(kbs).split(Fmt.NB)
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(p[0], style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, color = Fg)
            Text(p.getOrElse(1) { "" }, style = MaterialTheme.typography.labelSmall, color = Muted)
        }
    }
}

// ai: The buttons (the web's button, .primary, .text, .danger): pills 40 dp tall under a mouse; pad the sides' room.
enum class Kind { Primary, Tonal, Text, Danger }

@Composable
fun Btn(text: String, kind: Kind = Kind.Tonal, enabled: Boolean = true, modifier: Modifier = Modifier, pad: Dp = 20.dp, onClick: () -> Unit) {
    val mod = modifier.heightIn(min = 40.dp)
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

// ai: A choice among a few (Settings'): 36 dp, 8 dp corners, on the soft fill; the chosen one solid.
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

// ai: A row (the web's .item): a leading icon on a soft square, the title and the line under it, a trailing control;
// ai: the whole row its main action where it has one.
@Composable
fun ListRow(title: String, sub: String? = null, lead: ImageVector? = null, onClick: (() -> Unit)? = null, trailing: @Composable (() -> Unit)? = null) {
    Row(Modifier.fillMaxWidth().heightIn(min = if (sub != null) 64.dp else 56.dp)
        .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier).padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        if (lead != null) Box(Modifier.size(40.dp).clip(RoundedCornerShape(10.dp)).background(Soft), contentAlignment = Alignment.Center) {
            Icon(lead, contentDescription = null, Modifier.size(24.dp), tint = Fg)
        }
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = Fg, maxLines = 2, overflow = TextOverflow.Ellipsis)
            if (sub != null) Text(sub, style = MaterialTheme.typography.bodyMedium, color = Muted)
        }
        trailing?.invoke()
    }
}

// ai: The lab's text (the web's pre.readout): monospace on the card fill.
@Composable
fun CodeBlock(text: String) =
    Text(text, fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 17.sp, color = Fg,
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp).clip(RoundedCornerShape(8.dp)).background(Card).padding(horizontal = 12.dp, vertical = 8.dp))

// ai: A row that opens (the web's details.dev): a hairline over it, its title, a chevron that turns, and what it holds
// ai: straight under it; its state the caller's, kept (Prefs).
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

// ai: Settings' fields one rhythm (2026-10-02; the web's headers and spacing were inconsistent): 16 dp apart
// ai: (Fields), each its title over its control, the title row 36 dp whatever is in it (the value on the right, the
// ai: blocks' Auto chip at its end), a group's heading (Group) 8 dp over its first field.
@Composable
fun Fields(content: @Composable ColumnScope.() -> Unit) = Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(16.dp), content = content)

@Composable
fun Field(title: String, value: String = "", end: (@Composable () -> Unit)? = null, content: @Composable () -> Unit) {
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().heightIn(min = 36.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            // ai: the title on one line whatever the value's length; the value the rest of the row, cut at its end
            Text(title, style = MaterialTheme.typography.titleSmall, color = Muted, maxLines = 1, softWrap = false)
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

// ai: A slider in the web's look (2026-10-02): a 6 dp track
// ai: with round ends, filled up to a round thumb that covers the end it stands on. Material's own cut the track around
// ai: a bar thumb. choose: each value while dragged (the label live); done: on release, where the change is committed
// ai: (the web's sliders commit on release: a re-pick or a configure each step of a drag would drop frames for nothing).
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Bar(value: Float, range: ClosedFloatingPointRange<Float>, steps: Int, done: () -> Unit = {}, choose: (Float) -> Unit) {
    Slider(value, choose, modifier = Modifier.fillMaxWidth(), onValueChangeFinished = done, steps = steps, valueRange = range,
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
