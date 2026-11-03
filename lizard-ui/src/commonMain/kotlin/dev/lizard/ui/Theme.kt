package dev.lizard.ui

import androidx.compose.foundation.ScrollState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.compositeOver
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.lizard.readout.Readout
import dev.lizard.ui.res.Res
import dev.lizard.ui.res.inter_variable
import dev.lizard.ui.res.jetbrains_mono_regular
import org.jetbrains.compose.resources.Font

// ai: The look of the web's pages (lizard-web/ui.css; 2026-10-01), one for both apps (2026-10-31; each app's Parts.kt
// ai: before): light, black and white, flat; one solid black button on a screen, the others on a soft grey fill, minor
// ai: ones text alone, every button a pill; a top bar; cards on a soft fill. Colour only for an error and a finished
// ai: transfer.
val Fg = Color(0xFF111111)
val Bg = Color(0xFFFFFFFF)
val Muted = Color(0xFF666666)
val Line = Color(0xFFE4E4E4)
val Soft = Color(0xFFF1F1F1)
val Card = Color(0xFFF6F6F6)
// ai: a pressed button's grey: the Soft fill under the foreground's pressed state layer (10%), as a tonal Btn shows it
val Pressed = Fg.copy(alpha = 0.10f).compositeOver(Soft)
val Bad = Color(0xFFB00020)
// ai: a finished transfer (the web's --good, 2026-10-25 with the desktop's receiver): the received line and the rail's check
val Good = Color(0xFF0A7A2F)

fun toneColour(t: Readout.Tone) = when (t) { Readout.Tone.Bad -> Bad; Readout.Tone.Good -> Good; Readout.Tone.Plain -> Fg }

// ai: A target's least size, a button's, a field's and an icon button's: 48 dp under a finger, 40 under a mouse (the
// ai: web's ui.css, its pointer:fine rule); the app names its own (LizardTheme), the rest of the look is one.
val LocalTarget = staticCompositionLocalOf { 48.dp }

// ai: the pointer over an edge dragged sideways (SideEdge), the app's (LizardTheme's resizeIcon)
val LocalResizeIcon = staticCompositionLocalOf { PointerIcon.Default }

// ai: The faces (2026-10-10): the web's, from liblizard/vendor/fonts (build.gradle.kts fonts, the module's font
// ai: resources): Inter, one variable file read at each weight the type sets (regular, medium, semibold, bold) by its
// ai: axis, and JetBrains Mono for the readouts.
@OptIn(ExperimentalTextApi::class)
@Composable
private fun interFamily() = FontFamily(listOf(FontWeight.Normal, FontWeight.Medium, FontWeight.SemiBold, FontWeight.Bold).map { w ->
    Font(Res.font.inter_variable, w, variationSettings = FontVariation.Settings(FontVariation.weight(w.weight)))
})

val Mono: FontFamily
    @Composable get() = FontFamily(Font(Res.font.jetbrains_mono_regular))

// ai: Inter's bytes, for text a platform paints itself (the desktop sender's code area, AWT's)
suspend fun interBytes(): ByteArray = Res.readBytes("font/inter_variable.ttf")

// ai: The type: the web's sizes in sp, in Inter, and no letter spacing (Material's default runs 0.1 to 0.5 sp, which set
// ai: the text wider than the web's). Titles and labels semibold (a bar's title, a state line, a field's title, a
// ai: button's label), the small labels medium, the body regular.
private fun lizardType(inter: FontFamily): Typography {
    fun TextStyle.flat() = copy(letterSpacing = 0.sp, fontFamily = inter)
    val base = Typography()
    return Typography(
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
}

// ai: Material's surfaces all white (menus and dialogs float on a hairline, not a tint)
val LizardColours = lightColorScheme(primary = Fg, onPrimary = Bg, background = Bg, onBackground = Fg, surface = Bg, onSurface = Fg,
    surfaceVariant = Soft, onSurfaceVariant = Muted, outline = Line, outlineVariant = Line, error = Bad,
    surfaceContainerLowest = Bg, surfaceContainerLow = Bg, surfaceContainer = Bg, surfaceContainerHigh = Bg, surfaceContainerHighest = Bg)

// ai: The theme every screen of both apps is drawn in: the colours, the type in Inter, the app's target size and its
// ai: scrollbar beside a scrolling column (LocalScrollbar; none by default), and the pointer over a column's edge that
// ai: drags sideways (LocalResizeIcon; the platform's left-right resize cursor where it has one, the default otherwise)
@Composable
fun LizardTheme(target: Dp, scrollbar: @Composable (ScrollState, Modifier) -> Unit = { _, _ -> }, resizeIcon: PointerIcon = PointerIcon.Default,
                credits: List<Credit> = emptyList(), browse: (String) -> Unit = {}, content: @Composable () -> Unit) {
    val inter = interFamily()
    val type = remember(inter) { lizardType(inter) }
    CompositionLocalProvider(LocalTarget provides target, LocalScrollbar provides scrollbar, LocalResizeIcon provides resizeIcon, LocalCredits provides credits,
        LocalBrowse provides browse) {
        MaterialTheme(colorScheme = LizardColours, typography = type, content = content)
    }
}
