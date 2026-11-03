package dev.lizard.ui

import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import dev.lizard.readout.Readout
import java.text.DateFormat
import java.util.Date
import kotlin.math.floor

// ai: A scrolling column's bar where the platform shows one (the desktop's, beside its columns), drawn against the
// ai: column's own edge; nothing where it shows none (a phone's scroll shows its own)
val LocalScrollbar = staticCompositionLocalOf<@Composable (ScrollState, Modifier) -> Unit> { { _, _ -> } }

// ai: A screen (2026-10-02; Home's shape, 2026-10-31): the bar fixed at the top, under it one column at most 544 dp wide
// ai: (the web's 34rem), centred, scrolling on its own, the platform's scrollbar at the screen's edge
@Composable
fun Page(title: String?, onBack: (() -> Unit)?, content: @Composable ColumnScope.() -> Unit) {
    Column(Modifier.fillMaxSize().background(Bg).safeDrawingPadding()) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.TopCenter) {
            Box(Modifier.widthIn(max = 544.dp).fillMaxWidth().padding(horizontal = 16.dp)) { TopBar(title, onBack) }
        }
        Box(Modifier.weight(1f).fillMaxWidth()) {
            val scroll = rememberScrollState()
            Box(Modifier.fillMaxSize().verticalScroll(scroll), contentAlignment = Alignment.TopCenter) {
                Column(Modifier.widthIn(max = 544.dp).fillMaxWidth().padding(horizontal = 16.dp)) {
                    content()
                    Spacer(Modifier.height(32.dp))
                }
            }
            LocalScrollbar.current(scroll, Modifier.align(Alignment.CenterEnd).fillMaxHeight())
        }
    }
}

fun whenOf(at: Long): String = if (at > 0) DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(at)) else ""
fun countOf(n: Int) = "$n ${if (n == 1) "file" else "files"}"

// ai: A received file as Home lists it, or a bundle (several files or a folder sent as one, unpacked: a folder, its files'
// ai: count before its size): a press opens it; its menu the platform's actions, then Delete, asked first
class FileItem(val key: String, val name: String, val bundle: Boolean, val files: Int, val size: Long, val at: Long,
               val open: () -> Unit, val acts: List<Act>, val delete: () -> Unit)

// ai: What Home shows and does, the app's: its files and unfinished transfers, its words where a device names itself
interface HomeHost {
    val tipsSeen: Boolean
    fun tipsDone()
    fun tipsAgain()
    val tipSteps: List<String>
    fun openExample()
    val sendSolid: Boolean           // ai: the solid tile: Send on a desktop, Receive on a phone (as the web picks by pointer)
    fun goSend()
    fun goReceive()
    fun refresh()                    // ai: the lists read again, as Home shows
    val partials: List<Partial>
    fun liveFraction(p: Partial): Double?   // ai: the share in where the receiver holds that transfer now
    fun deletePartial(p: Partial)
    val files: List<FileItem>
    fun deleteAll()
    val note: String                 // ai: what came of a Save or a filing; "" none
    val device: String               // ai: "computer", "phone": the words of a deletion
    val about: String
}

// ai: Home (the web's index.html; both apps' since 2026-10-31): the first-run tip (its first line semibold) until it is dismissed, Send and Receive
// ai: side by side, the unfinished transfers (shown where there are any: a press or Continue opens Receive, which
// ai: resumes one once its file is in the light), the received files (shown where there are any), and Settings (the
// ai: tips again, the received files with Delete all, About).
@Composable
fun HomeScreen(h: HomeHost) {
    LaunchedEffect(Unit) { h.refresh() }
    Page(null, null) {
        if (!h.tipsSeen) TipCard(onGot = { h.tipsDone() }) {
            Text("Send a file from one screen to another device's camera.", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.SemiBold, color = Fg)
            Steps(h.tipSteps)
            Text("An example on video", style = MaterialTheme.typography.bodyLarge.copy(textDecoration = TextDecoration.Underline), color = Fg,
                modifier = Modifier.pointerHoverIcon(PointerIcon.Hand).clickable(role = Role.Button) { h.openExample() })
        }
        Row(Modifier.fillMaxWidth().padding(top = 16.dp, bottom = 8.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            HomeTile(Icons.monitor, "Send", solid = h.sendSolid, modifier = Modifier.weight(1f)) { h.goSend() }
            HomeTile(Icons.camera, "Receive", solid = !h.sendSolid, modifier = Modifier.weight(1f)) { h.goReceive() }
        }
        val partials = h.partials
        if (partials.isNotEmpty()) {
            SectionLabel("Unfinished")
            partials.forEachIndexed { i, p -> key(p.id) { PartialRow(h, p, last = i == partials.lastIndex) } }
        }
        val files = h.files
        if (files.isNotEmpty()) SectionLabel("Received files")
        // ai: keyed by the file, so a row's menu and its question stay with it when a file is filed above it meanwhile
        files.forEachIndexed { i, e -> key(e.key) { FileRow(h, e, last = i == files.lastIndex) } }
        h.note.let { if (it.isNotEmpty()) Text(it, style = MaterialTheme.typography.bodyMedium, color = if (it.startsWith("Saved")) Muted else Bad, modifier = Modifier.padding(top = 8.dp)) }
        HomeSettings(h)
    }
}

// ai: A card of Home's (the web's .go .btn): its icon over its title, centred in the card (2026-11-02, the line under the
// ai: title gone, the card 104 dp tall at least, 120 with it); the whole card the action; the solid one the foreground's fill
@Composable
private fun HomeTile(icon: ImageVector, title: String, solid: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val fg = if (solid) Bg else Fg
    Column(modifier.heightIn(min = 104.dp).clip(RoundedCornerShape(16.dp)).background(if (solid) Fg else Soft)
        .clickable(role = Role.Button, onClick = onClick).padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp, Alignment.CenterVertically)) {
        Icon(icon, contentDescription = null, Modifier.size(24.dp), tint = fg)
        Spacer(Modifier.height(8.dp))
        Text(title, style = MaterialTheme.typography.titleMedium, color = fg)
    }
}

// ai: An unfinished transfer's row: its name (or "a file") and "37% of 3.1 MB · date" (the receiver's live share where
// ai: it holds that root); a press opens Receive with the camera, its menu Continue (the same) and Delete, asked first
@Composable
private fun PartialRow(h: HomeHost, p: Partial, last: Boolean) {
    var asking by remember { mutableStateOf(false) }
    val frac = h.liveFraction(p) ?: p.fraction
    val name = p.name.ifEmpty { "a file" }
    ListRow(name, listOf("${floor(100 * frac).toInt()}% of ${Readout.bytes(p.length)}", whenOf(p.at)).filter { it.isNotEmpty() }.joinToString(" · "),
        lead = Icons.file, divider = !last, onClick = { h.goReceive() }) {
        MoreMenu("More for $name", listOf(Act("Continue") { h.goReceive() }, Act("Delete", danger = true) { asking = true }))
    }
    if (asking) AskDialog("Delete $name?", "The part received goes from this ${h.device}.", "Delete", { asking = false }) { h.deletePartial(p) }
}

@Composable
private fun FileRow(h: HomeHost, e: FileItem, last: Boolean) {
    var asking by remember { mutableStateOf(false) }
    ListRow(e.name, listOf(if (e.bundle) countOf(e.files) else "", Readout.bytes(e.size), whenOf(e.at)).filter { it.isNotEmpty() }.joinToString(" · "),
        lead = if (e.bundle) Icons.folder else Icons.file, divider = !last, onClick = e.open) {
        MoreMenu("More for ${e.name}", e.acts + Act("Delete", danger = true) { asking = true })
    }
    if (asking) AskDialog("Delete ${e.name}?", "It goes from this ${h.device}." + if (h.device == "phone") " A copy saved elsewhere stays." else "",
        "Delete", { asking = false }) { e.delete() }
}

// ai: Home's settings (the web Home's .settings): the tips again, the received files' count and size with Delete all
// ai: (asked first; the unfinished transfers stay), About (a fold: the build line and the credits, AboutFold)
@Composable
private fun HomeSettings(h: HomeHost) {
    var asking by remember { mutableStateOf(false) }
    val files = h.files
    val n = files.size
    SectionLabel("Settings")
    ListRow("Show tips again", divider = true, onClick = { h.tipsAgain() })
    ListRow("Received files", if (n == 0) "No files kept." else "${countOf(n)} kept, ${Readout.bytes(files.sumOf { it.size })}") {
        if (n > 0) Btn("Delete all", Kind.Danger) { asking = true }
    }
    AboutFold(h.about)
    if (asking) AskDialog("Delete ${countOf(n)}?", "They go from this ${h.device}." + if (h.device == "phone") " Copies saved elsewhere stay." else "",
        "Delete all", { asking = false }) { h.deleteAll() }
}

// ai: A bundle's files (2026-10-16; where a platform opens a bundle in the app, the phone): its count and size, its
// ai: actions (Share all of them, Save the folder into one picked), then a row a file, its path in the folder and its
// ai: size, a press opening it, its menu the file's actions
class BundleFile(val path: String, val size: Long, val open: () -> Unit, val acts: List<Act>)

@Composable
fun BundleScreen(name: String, files: Int, size: Long, acts: List<Act>, items: List<BundleFile>, note: String, onBack: () -> Unit) {
    Page(name, onBack) {
        Text("${countOf(files)} · ${Readout.bytes(size)}", style = MaterialTheme.typography.bodyMedium, color = Muted, modifier = Modifier.padding(top = 8.dp))
        Row(Modifier.fillMaxWidth().padding(vertical = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (a in acts) Btn(a.label, modifier = Modifier.weight(1f)) { a.run() }
        }
        if (note.isNotEmpty()) Text(note, style = MaterialTheme.typography.bodyMedium, color = if (note.startsWith("Saved")) Muted else Bad, modifier = Modifier.padding(bottom = 8.dp))
        items.forEachIndexed { i, it -> key(it.path) {
            ListRow(it.path, Readout.bytes(it.size), lead = Icons.file, divider = i != items.lastIndex, onClick = it.open) { MoreMenu("More for ${it.path}", it.acts) }
        } }
    }
}
