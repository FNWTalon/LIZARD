package dev.lizard.ui

import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

// ai: A fold's content as an app describes it (2026-10-31): each setting what it shows and what a change does, the app
// ai: deciding which there are and their order (a camera's own controls, a platform's own fields), the shared screens
// ai: drawing them in the fields' rhythm (Fields).
sealed interface Setting

// ai: a menu (Select)
class Choice<T>(val title: String, val value: T, val options: List<Pair<T, String>>, val blank: String = "", val enabled: Boolean = true,
                val enabledFor: (T) -> Boolean = { true }, val onOpen: () -> Unit = {}, val choose: (T) -> Unit) : Setting

// ai: a choice among a few as chips under its title (Chips: Send's FPS, Ring, Codes, Encoder, VSync, Payload)
class ChipChoice<T>(val title: String, val value: T, val options: List<Pair<T, String>>, val enabled: Boolean = true, val choose: (T) -> Unit) : Setting

// ai: a slider (Bar) under its title and the value as it reads: each value while dragged, `done` at the release
class Slide(val title: String, val label: String, val value: Float, val range: ClosedFloatingPointRange<Float>, val steps: Int,
            val done: (() -> Unit)? = null, val choose: (Float) -> Unit) : Setting

// ai: a line of text (TextInput), done at Enter or as the focus leaves it changed
class Typed(val title: String, val value: String, val placeholder: String, val done: (String) -> Unit) : Setting

// ai: a field of lines (the PC link's: its state on the right, its lines muted under it, one action under them)
class Lines(val title: String, val value: String, val lines: List<String>, val act: Act? = null) : Setting

// ai: the lab's text (CodeBlock); a button across the fold (a run's Download, Show in folder)
class Code(val text: String, val size: Int = 13) : Setting
class Press(val act: Act, val enabled: Boolean = true) : Setting

@Composable
fun Settings(items: List<Setting>) = Fields { for (s in items) SettingView(s) }

@Composable
private fun ColumnScope.SettingView(s: Setting) {
    when (s) {
        is Choice<*> -> ChoiceView(s)
        is ChipChoice<*> -> ChipsView(s)
        is Slide -> Field(s.title, s.label) { Bar(s.value, s.range, s.steps, s.done, s.choose) }
        is Typed -> Field(s.title) { TextInput(s.value, s.placeholder, s.done) }
        is Lines -> Field(s.title, s.value) {
            for (l in s.lines) Text(l, style = MaterialTheme.typography.bodyMedium, color = Muted)
            s.act?.let { a -> Btn(a.label, modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) { a.run() } }
        }
        is Code -> CodeBlock(s.text, s.size)
        is Press -> Btn(s.act.label, enabled = s.enabled, modifier = Modifier.fillMaxWidth()) { s.act.run() }
    }
}

@Composable
private fun <T> ChoiceView(c: Choice<T>) =
    Select(c.title, c.value, c.options, c.blank, c.enabled, c.enabledFor, c.onOpen, c.choose)

@Composable
private fun <T> ChipsView(c: ChipChoice<T>) = Field(c.title) {
    Chips { for ((k, label) in c.options) Chip(label, k == c.value, c.enabled) { if (k != c.value) c.choose(k) } }
}

// ai: A count's slider whose leftmost step is auto (send.html #blocks and #subchAuto, 2026-10-05): the setting `value` (0
// ai: auto), `shown` what auto takes now (0 while unknown), `range` the counts by hand; its value "Auto, <text>" while
// ai: what auto takes is known (`auto` names the leftmost step: the phone's Brightness "System")
fun autoSlide(title: String, value: Int, shown: Int, range: IntRange, text: (Int) -> String, done: (() -> Unit)?, auto: String = "Auto",
              choose: (Int) -> Unit): Slide {
    val isAuto = value == 0
    val b = if (isAuto) shown else value
    val lo = range.first - 1
    return Slide(title, if (isAuto) (if (b > 0) "$auto, ${text(b)}" else auto) else text(b), (if (isAuto) lo else value).toFloat(),
        lo.toFloat()..range.last.toFloat(), range.last - lo - 1, done) { val v = it.roundToInt(); choose(if (v < range.first) 0 else v) }
}
