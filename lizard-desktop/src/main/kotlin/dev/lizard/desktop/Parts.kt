package dev.lizard.desktop

import androidx.compose.foundation.BorderStroke
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
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
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
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.platform.Font
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.lizard.readout.Readout
import kotlin.math.floor

// ai: The look of the web's pages and the Android app (lizard-web/ui.css; lizard-android/.../Parts.kt, ported without Android
// ai: APIs; 2026-10-01): light, black and white, flat; one solid black button on a screen, the others on a soft grey
// ai: fill, minor ones text alone, every button a pill;
// ai: a top bar; cards on a soft fill. Colour only for an error and a finished transfer (Good, since the receiver,
// ai: 2026-10-25). The sizes are the web's under a mouse (ui.css pointer:fine): buttons, fields and icon targets 40 dp,
// ai: where the phone's are 48.
val Fg = Color(0xFF111111)
val Bg = Color(0xFFFFFFFF)
val Muted = Color(0xFF666666)
val Line = Color(0xFFE4E4E4)
val Soft = Color(0xFFF1F1F1)
val Card = Color(0xFFF6F6F6)
val Bad = Color(0xFFB00020)
// ai: a finished transfer (the web's --good, 2026-10-25 with the receiver): the received line and the rail's check
val Good = Color(0xFF0A7A2F)

// ai: The faces (2026-10-10): the web's, from liblizard/vendor/fonts (build.gradle.kts fonts): Inter, one variable file
// ai: read at each weight the app sets by its axis, and JetBrains Mono for the readouts.
@OptIn(ExperimentalTextApi::class)
private fun inter(w: FontWeight) = Font("fonts/inter/InterVariable.ttf", w, variationSettings = FontVariation.Settings(FontVariation.weight(w.weight)))
val Inter = FontFamily(listOf(FontWeight.Normal, FontWeight.Medium, FontWeight.SemiBold, FontWeight.Bold).map(::inter))
val Mono = FontFamily(Font("fonts/jetbrains-mono/JetBrainsMono-Regular.ttf"))

// ai: The type: the web's sizes in sp, in Inter, and no letter spacing (Material's default runs 0.1 to 0.5 sp).
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
// ai: Material's surfaces all white (menus float on a hairline, not a tint)
val LizardColours = lightColorScheme(primary = Fg, onPrimary = Bg, background = Bg, onBackground = Fg, surface = Bg, onSurface = Fg,
    surfaceVariant = Soft, onSurfaceVariant = Muted, outline = Line, outlineVariant = Line, error = Bad,
    surfaceContainerLowest = Bg, surfaceContainerLow = Bg, surfaceContainer = Bg, surfaceContainerHigh = Bg, surfaceContainerHighest = Bg)

// ai: The icons: Feather's (MIT) on a 24 grid, 2 px strokes, as the web pages and the app draw them; and LIZARD's mark
// ai: (lizard-web/icon.svg: the corner marks, the dark ring between them, a grey picture, on white), the window's icon.
object Icons {
    private fun stroked(name: String, vararg d: String) = ImageVector.Builder(name, 24.dp, 24.dp, 24f, 24f).apply {
        for (p in d) addPath(addPathNodes(p), stroke = SolidColor(Fg), strokeLineWidth = 2f, strokeLineCap = StrokeCap.Round, strokeLineJoin = StrokeJoin.Round)
    }.build()
    // ai: a circle as a path (Feather draws some of its icons with <circle>)
    private fun circle(cx: Float, cy: Float, r: Float) = "M${cx - r} ${cy}a$r $r 0 1 0 ${2 * r} 0a$r $r 0 1 0 ${-2 * r} 0"
    val chevron = stroked("chevron", "M6,9l6,6 6,-6")
    val collapse = stroked("collapse", "M11,17l-5,-5 5,-5M18,17l-5,-5 5,-5")
    val play = stroked("play", "M6,3l14,9 -14,9z")
    val pause = stroked("pause", "M6,4h4v16h-4z M14,4h4v16h-4z")
    val file = stroked("file", "M14,2H6a2,2 0,0 0,-2 2v16a2,2 0,0 0,2 2h12a2,2 0,0 0,2 -2V8z", "M14,2v6h6")
    // ai: the receiver's and Home's (2026-10-25), the web pages' paths (lizard-web/recv.html, index.html, ui.mjs ICONS)
    val swap = stroked("swap", "M8 3L4 7l4 4M4 7h16M16 21l4-4-4-4M20 17H4")
    val camera = stroked("camera", "M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z", circle(12f, 13f, 4f))
    val cameraOff = stroked("camera-off", "M1 1l22 22M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34m-7.72-2.06a4 4 0 1 1-5.56-5.56")
    val monitor = stroked("monitor", "M4 3h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z", "M8 21h8M12 17v4")
    val check = stroked("check", "M22 11.08V12a10 10 0 1 1-5.93-9.14", "M22 4L12 14.01l-3-3")
    val open = stroked("open", "M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6", "M15 3h6v6", "M10 14L21 3")
    val folder = stroked("folder", "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z")
    val more = stroked("more", circle(12f, 5f, 1f), circle(12f, 12f, 1f), circle(12f, 19f, 1f))
    val info = stroked("info", circle(12f, 12f, 10f), "M12 16v-4M12 8h.01")
    val back = stroked("back", "M19 12H5M12 19l-7-7 7-7")
    val mark = ImageVector.Builder("mark", 64.dp, 64.dp, 64f, 64f).apply {
        addPath(addPathNodes("M0 0h64v64h-64z"), fill = SolidColor(Color.White))
        // ai: the ink spans 70% of the square (2026-10-10, the standard lizard-web/icon.svg states)
        addGroup(pivotX = 32f, pivotY = 32f, scaleX = 1.24444f, scaleY = 1.24444f)
        addPath(addPathNodes("M14 14h9v9h-9zM41 14h9v9h-9zM14 41h9v9h-9zM41 41h9v9h-9zM25 16h14v3H25zM25 45h14v3H25zM16 25h3v14h-3zM45 25h3v14h-3z"), fill = SolidColor(Fg))
        addPath(addPathNodes("M25 25h7v7h-7zM32 32h7v7h-7z"), fill = SolidColor(Color(0xFF777777)))
        addPath(addPathNodes("M32 25h7v7h-7zM25 32h7v7h-7z"), fill = SolidColor(Color(0xFFBBBBBB)))
        clearGroup()
    }.build()
}

// ai: A line that names a file (2026-10-23; the web's ui.mjs sayName, the Android app's NameLine): its head, up to the
// ai: name and a comma after it, on one line, the name's stem cut with an ellipsis where the column is too narrow and
// ai: its ending kept whole (Readout.nameParts); the rest after it there where it fits, else under it.
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun NameLine(before: String, name: String, after: String, style: TextStyle, color: Color, modifier: Modifier = Modifier) {
    val (stem, end) = Readout.nameParts(name)
    val comma = after.startsWith(",")
    val rest = (if (comma) after.drop(1) else after).trim()
    FlowRow(modifier, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        Row {
            Text(before + stem, style = style, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
            Text(end + if (comma) "," else "", style = style, color = color, maxLines = 1)
        }
        if (rest.isNotEmpty()) Text(rest, style = style, color = color)
    }
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
fun IconBtn(icon: ImageVector, label: String, enabled: Boolean = true, turn: Float = 0f, onClick: () -> Unit) {
    Box(Modifier.size(40.dp).clip(CircleShape).clickable(enabled = enabled, role = Role.Button, onClick = onClick).semantics { contentDescription = label },
        contentAlignment = Alignment.Center) {
        Icon(icon, contentDescription = null, Modifier.size(24.dp).rotate(turn), tint = if (enabled) Fg else Fg.copy(alpha = .38f))
    }
}

// ai: The sidebar collapser (the web's #collapse, 2026-10-02): chevrons pointing to where the column goes.
@Composable
fun CollapseBtn(collapsed: Boolean, onClick: () -> Unit) =
    IconBtn(Icons.collapse, if (collapsed) "Show sidebar" else "Hide sidebar", turn = if (collapsed) 180f else 0f, onClick = onClick)

// ai: The collapsed column (the web's 56 px rail): the collapser at its top, a hairline under its row as the bar's, then
// ai: the caller's squares; `foot` at its bottom (Home, 2026-10-25: the web's rail's last square, the only way home from
// ai: a tool page in landscape).
@Composable
fun Rail(onClick: () -> Unit, foot: @Composable () -> Unit = {}, squares: @Composable () -> Unit = {}) {
    Column(Modifier.width(56.dp).fillMaxHeight(), horizontalAlignment = Alignment.CenterHorizontally) {
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
        val p = Readout.rate(kbs).split('\u00a0')
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(p[0], style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, color = Fg)
            Text(p.getOrElse(1) { "" }, style = MaterialTheme.typography.labelSmall, color = Muted)
        }
    }
}

// ai: The share of a file in, in a rail's square (the receiver's, the Android app's PctSquare): its whole percent over
// ai: "%", as the rate's figure over its unit
@Composable
fun PctSquare(frac: Double) = Square {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text("${floor(100 * frac).toInt()}", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, color = Fg)
        Text("%", style = MaterialTheme.typography.labelSmall, color = Muted)
    }
}

// ai: The other tool, at the bar's end (the web's #swap, 2026-10-25 on the desktop): the swap arrows and its name, a text
// ai: button 40 dp tall under a mouse (the phone's 48).
@Composable
fun SwapBtn(label: String, onClick: () -> Unit) =
    TextButton(onClick, Modifier.heightIn(min = 40.dp), shape = CircleShape, colors = ButtonDefaults.textButtonColors(contentColor = Fg),
        contentPadding = PaddingValues(horizontal = 12.dp)) {
        Icon(Icons.swap, contentDescription = null, Modifier.size(24.dp), tint = Fg)
        Spacer(Modifier.width(8.dp))
        Text(label, style = MaterialTheme.typography.labelLarge, maxLines = 1, softWrap = false)
    }

// ai: A file's progress (the web's .meter): a 6 dp track on the soft fill, round ends, filled in the foreground's colour
// ai: to frac; nothing for null
@Composable
fun Meter(frac: Double?) {
    if (frac == null) return
    Box(Modifier.fillMaxWidth().padding(vertical = 8.dp).height(6.dp).clip(RoundedCornerShape(3.dp)).background(Soft)) {
        if (frac > 0) Box(Modifier.fillMaxWidth(frac.toFloat().coerceIn(0f, 1f)).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(Fg))
    }
}

// ai: The web's spinner (recv.html #loading .spin): a 28 dp ring of 3 dp in the hairline's grey, its turning arc the
// ai: foreground's
@Composable
fun Spinner() = CircularProgressIndicator(Modifier.size(28.dp), color = Fg, strokeWidth = 3.dp, trackColor = Line)

// ai: A choice among several (the web's select, 2026-10-25 on the desktop; Settings' chips everywhere until then): a
// ai: field's title over a 40 dp field on the soft fill, 8 dp corners, the chosen option's label and a chevron, a click
// ai: opening the options as a menu as wide as the field, the chosen one on the soft fill (as a native select's open
// ai: list marks it). `blank` shows where no option is the value; `onOpen` runs as the menu opens (a list read again
// ai: then).
@Composable
fun <T> Select(title: String, value: T, options: List<Pair<T, String>>, blank: String = "", onOpen: () -> Unit = {}, onChoose: (T) -> Unit) {
    var open by remember { mutableStateOf(false) }
    var width by remember { mutableStateOf(0) }
    val density = LocalDensity.current
    Field(title) {
        Box(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth().heightIn(min = 40.dp).onSizeChanged { width = it.width }.clip(RoundedCornerShape(8.dp)).background(Soft)
                .clickable(role = Role.Button) { onOpen(); open = true }.padding(start = 12.dp, end = 12.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Text(options.firstOrNull { it.first == value }?.second ?: blank, style = MaterialTheme.typography.bodyLarge,
                    color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                Icon(Icons.chevron, contentDescription = null, Modifier.size(18.dp), tint = Muted)
            }
            DropdownMenu(open, { open = false }, Modifier.width(with(density) { width.toDp() }).heightIn(max = 480.dp),
                shape = RoundedCornerShape(12.dp), containerColor = Bg, shadowElevation = 0.dp, border = BorderStroke(1.dp, Line)) {
                for ((k, label) in options) DropdownMenuItem({ Text(label, style = MaterialTheme.typography.bodyLarge, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                    { open = false; if (k != value) onChoose(k) }, if (k == value) Modifier.background(Soft) else Modifier,
                    colors = MenuDefaults.itemColors(textColor = Fg))
            }
        }
    }
}

// ai: A line of text in the select's look (2026-10-27, the phone's address): the soft fill, 8 dp corners, the
// ai: placeholder muted while empty; done with the text at Enter, or as the focus leaves it changed
@Composable
fun TextInput(value: String, placeholder: String, done: (String) -> Unit) {
    var text by remember(value) { mutableStateOf(value) }
    BasicTextField(text, { text = it }, singleLine = true, textStyle = MaterialTheme.typography.bodyLarge.copy(color = Fg), cursorBrush = SolidColor(Fg),
        modifier = Modifier.fillMaxWidth().heightIn(min = 40.dp).clip(RoundedCornerShape(8.dp)).background(Soft)
            .onPreviewKeyEvent { e -> if (e.key == Key.Enter && e.type == KeyEventType.KeyUp) { done(text.trim()); true } else false }
            .onFocusChanged { if (!it.isFocused && text.trim() != value) done(text.trim()) },
        decorationBox = { inner ->
            Box(Modifier.padding(horizontal = 12.dp).heightIn(min = 40.dp), contentAlignment = Alignment.CenterStart) {
                if (text.isEmpty()) Text(placeholder, style = MaterialTheme.typography.bodyLarge, color = Muted)
                inner()
            }
        })
}

// ai: A row's more menu (the web's .menu: the more icon, its actions in a list on white with a hairline round it, 12 dp
// ai: corners, no shadow, its right edge under the icon's): each action (its label, whether it is a deletion's, red,
// ai: and what it does)
class Act(val label: String, val danger: Boolean = false, val run: () -> Unit)

@Composable
fun MoreMenu(label: String, acts: List<Act>) {
    var open by remember { mutableStateOf(false) }
    Box {
        IconBtn(Icons.more, label) { open = true }
        DropdownMenu(open, { open = false }, Modifier.widthIn(min = 176.dp), offset = DpOffset((40 - 176).dp, 4.dp), shape = RoundedCornerShape(12.dp),
            containerColor = Bg, shadowElevation = 0.dp, border = BorderStroke(1.dp, Line)) {
            for (a in acts) DropdownMenuItem({ Text(a.label, style = MaterialTheme.typography.bodyLarge) }, { open = false; a.run() },
                colors = MenuDefaults.itemColors(textColor = if (a.danger) Bad else Fg))
        }
    }
}

// ai: A section's heading (the web's section h2): small, grey, 24 dp over it
@Composable
fun SectionLabel(text: String) =
    Text(text, style = MaterialTheme.typography.titleSmall, color = Muted, modifier = Modifier.padding(top = 24.dp, bottom = 4.dp))

// ai: A question before a deletion (the Android app's AlertDialog, the web's "Delete it?" asked on the button): its title,
// ai: what goes, Cancel and the red action; on white
@Composable
fun AskDialog(title: String, text: String, action: String, onDismiss: () -> Unit, onConfirm: () -> Unit) =
    AlertDialog(onDismissRequest = onDismiss,
        confirmButton = { Btn(action, Kind.Danger) { onDismiss(); onConfirm() } },
        dismissButton = { Btn("Cancel", Kind.Text, onClick = onDismiss) },
        title = { Text(title, style = MaterialTheme.typography.titleMedium, color = Fg) },
        text = { Text(text, style = MaterialTheme.typography.bodyLarge, color = Fg) },
        shape = RoundedCornerShape(12.dp), containerColor = Bg)

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
// ai: the whole row its main action where it has one; a hairline under it where `divider` (the web's rows of a list).
// ai: The title one line, cut with an ellipsis (2026-10-25, as the web's and the app's since 2026-10-23; two lines before).
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

// ai: The lab's text (the web's pre.readout): JetBrains Mono on the card fill; at 13 sp the web's .lab line, muted (the
// ai: receiver's lab line, 2026-10-25, as the Android app's CodeBlock), at 12 the readout in the foreground's colour.
@Composable
fun CodeBlock(text: String, size: Int = 12) =
    Text(text, fontFamily = Mono, fontSize = size.sp, lineHeight = (size + 5).sp, color = if (size < 13) Fg else Muted,
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
// ai: (Fields), each its title over its control, the title row 36 dp whatever is in it (the value on the right), a
// ai: group's heading (Group) 8 dp over its first field.
@Composable
fun Fields(content: @Composable ColumnScope.() -> Unit) = Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(16.dp), content = content)

@Composable
fun Field(title: String, value: String = "", content: @Composable () -> Unit) {
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().heightIn(min = 36.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            // ai: the title on one line whatever the value's length; the value the rest of the row, cut at its end
            Text(title, style = MaterialTheme.typography.titleSmall, color = Muted, maxLines = 1, softWrap = false)
            Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
                if (value.isNotEmpty()) Text(value, style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
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
