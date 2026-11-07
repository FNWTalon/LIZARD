package dev.lizard.ui

import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.material3.HorizontalDivider
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.TextButton
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.sp
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
import androidx.compose.ui.semantics.Role
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

// ai: What Home shows and does, the app's: its files and unfinished transfers
interface HomeHost {
    val sendSolid: Boolean           // ai: the solid tile: Send on a desktop, Receive on a phone (as the web picks by pointer)
    fun goSend()
    fun goReceive()
    fun refresh()                    // ai: the lists read again, as Home shows
    val partials: List<Partial>
    fun liveFraction(p: Partial): Double?   // ai: the share in where the receiver holds that transfer now
    fun deletePartial(p: Partial)
    val files: List<FileItem>
    fun deleteAll()
    val note: String                 // ai: a Save's or a filing's failure, in red; "" none (a Save that works says nothing, 2026-11-06)
    val about: String
}

// ai: Home (the web's index.html; both apps' since 2026-10-31): Send and Receive side by side, the unfinished transfers
// ai: (shown where there are any: a press or Continue opens Receive, which resumes one once its file is in the light),
// ai: the received files (shown where there are any), and Settings (the received files with Delete all, About). No
// ai: tips (2026-11-03).
@Composable
fun HomeScreen(h: HomeHost) = BoxWithConstraints {
    LaunchedEffect(Unit) { h.refresh() }
    // ai: a PC's window (a mouse's target, 900 dp wide or more): the spacious layout (HomeWide, 2026-11-06)
    if (maxWidth >= 900.dp && LocalTarget.current < 48.dp) HomeWide(h) else HomeColumn(h)
}

@Composable
private fun HomeColumn(h: HomeHost) {
    Page(null, null) {
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
        h.note.let { if (it.isNotEmpty()) Text(it, style = MaterialTheme.typography.bodyMedium, color = Bad, modifier = Modifier.padding(top = 8.dp)) }
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
    if (asking) AskDialog("Delete $name?", "Delete", { asking = false }) { h.deletePartial(p) }
}

@Composable
private fun FileRow(h: HomeHost, e: FileItem, last: Boolean) {
    var asking by remember { mutableStateOf(false) }
    ListRow(e.name, listOf(if (e.bundle) countOf(e.files) else "", Readout.bytes(e.size), whenOf(e.at)).filter { it.isNotEmpty() }.joinToString(" · "),
        lead = if (e.bundle) Icons.folder else Icons.file, divider = !last, onClick = e.open) {
        MoreMenu("More for ${e.name}", e.acts + Act("Delete", danger = true) { asking = true })
    }
    if (asking) AskDialog("Delete ${e.name}?", "Delete", { asking = false }) { e.delete() }
}

// ai: Home's settings (the web Home's .settings): the received files' count and size with Delete all
// ai: (asked first; the unfinished transfers stay), About (a fold: the build line and the credits, AboutFold)
@Composable
private fun HomeSettings(h: HomeHost) {
    var asking by remember { mutableStateOf(false) }
    val files = h.files
    val n = files.size
    SectionLabel("Settings")
    ListRow("Received files", if (n == 0) "None" else "${countOf(n)} · ${Readout.bytes(files.sumOf { it.size })}") {
        if (n > 0) Btn("Delete all", Kind.Danger) { asking = true }
    }
    AboutFold(h.about)
    SourceRow()
    if (asking) AskDialog("Delete ${countOf(n)}?", "Delete all", { asking = false }) { h.deleteAll() }
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
        if (note.isNotEmpty()) Text(note, style = MaterialTheme.typography.bodyMedium, color = Bad, modifier = Modifier.padding(bottom = 8.dp))
        items.forEachIndexed { i, it -> key(it.path) {
            ListRow(it.path, Readout.bytes(it.size), lead = Icons.file, divider = i != items.lastIndex, onClick = it.open) { MoreMenu("More for ${it.path}", it.acts) }
        } }
    }
}


// ai: Home on a PC (2026-11-06; the web's index.html the same), spacious, the one place a PC's buttons stretch: the bar
// ai: across the window (the mark and LIZARD; About, the credits in a dialog, and Source code on its right), the content
// ai: at most 1120 dp wide with 48 dp margins; Send and Receive two cards stretched across it, 200 dp tall; Unfinished
// ai: and Received files as lists with columns (Name, Progress or Size, Started or Received), each row's Continue or
// ai: Open beside its menu, Delete all on the files' heading; no Settings section.
private val WIDE = 1120.dp
private val PAD = 48.dp

@Composable
private fun HomeWide(h: HomeHost) {
    var about by remember { mutableStateOf(false) }
    var asking by remember { mutableStateOf(false) }
    val browse = LocalBrowse.current
    val files = h.files
    val partials = h.partials
    Column(Modifier.fillMaxSize().background(Bg).safeDrawingPadding()) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.TopCenter) {
            Row(Modifier.widthIn(max = WIDE).fillMaxWidth().height(64.dp).padding(horizontal = PAD), verticalAlignment = Alignment.CenterVertically) {
                Image(Icons.mark, contentDescription = null, Modifier.size(28.dp))
                Spacer(Modifier.size(10.dp))
                Text("LIZARD", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = Fg, modifier = Modifier.weight(1f))
                Row(Modifier.offset(x = 12.dp)) {
                    QuietBtn("About") { about = true }
                    QuietBtn("Source code") { browse(SOURCE_URL) }
                }
            }
        }
        HorizontalDivider(color = Line)
        Box(Modifier.weight(1f).fillMaxWidth()) {
            val scroll = rememberScrollState()
            Box(Modifier.fillMaxSize().verticalScroll(scroll), contentAlignment = Alignment.TopCenter) {
                Column(Modifier.widthIn(max = WIDE).fillMaxWidth().padding(horizontal = PAD)) {
                    Spacer(Modifier.height(48.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                        BigTile(Icons.monitor, "Send", solid = h.sendSolid, modifier = Modifier.weight(1f)) { h.goSend() }
                        BigTile(Icons.camera, "Receive", solid = !h.sendSolid, modifier = Modifier.weight(1f)) { h.goReceive() }
                    }
                    Spacer(Modifier.height(56.dp))
                    if (partials.isNotEmpty()) {
                        Heading("Unfinished")
                        Columns("Progress", "Started")
                        for (p in partials) key(p.id) { PartialLine(h, p) }
                        Spacer(Modifier.height(48.dp))
                    }
                    // ai: the section stays with no files (2026-11-06): its empty box in place of the columns
                    if (files.isNotEmpty()) {
                        Heading("Received files") { Btn("Delete all", Kind.Danger, modifier = Modifier.offset(x = 12.dp)) { asking = true } }
                        Columns("Size", "Received")
                        for (e in files) key(e.key) { FileLine(e) }
                    } else {
                        Heading("Received files")
                        EmptyBox("No received files")
                    }
                    h.note.let { if (it.isNotEmpty()) Text(it, style = MaterialTheme.typography.bodyMedium, color = Bad, modifier = Modifier.padding(top = 12.dp)) }
                    Spacer(Modifier.height(48.dp))
                }
            }
            LocalScrollbar.current(scroll, Modifier.align(Alignment.CenterEnd).fillMaxHeight())
        }
    }
    if (about) AboutDialog(h.about) { about = false }
    if (asking) AskDialog("Delete ${countOf(files.size)}?", "Delete all", { asking = false }) { h.deleteAll() }
}

// ai: an empty list's box (the web's #emptyPc): 200 dp tall, a 2 dp dashed outline in a mid grey, 16 dp corners, the inbox
// ai: icon over the words, muted
@Composable
private fun EmptyBox(text: String) {
    val dash = Color(0xFFBBBBBB)
    Column(Modifier.fillMaxWidth().height(200.dp).drawBehind {
        val w = 2.dp.toPx()
        drawRoundRect(dash, topLeft = Offset(w / 2, w / 2), size = Size(size.width - w, size.height - w), cornerRadius = CornerRadius(16.dp.toPx()),
            style = Stroke(width = w, pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx()))))
    }, horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically)) {
        Icon(Icons.inbox, contentDescription = null, Modifier.size(36.dp), tint = Muted)
        Text(text, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Medium, color = Muted)
    }
}

// ai: the bar's quiet text buttons (About, Source code): muted, medium
@Composable
private fun QuietBtn(text: String, onClick: () -> Unit) =
    TextButton(onClick, Modifier.heightIn(min = LocalTarget.current), shape = CircleShape, colors = ButtonDefaults.textButtonColors(contentColor = Muted),
        contentPadding = PaddingValues(horizontal = 12.dp)) { Text(text, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Medium) }

// ai: Send or Receive on a PC: a card 200 dp tall, its 48 dp icon over its title (22 sp)
@Composable
private fun BigTile(icon: ImageVector, title: String, solid: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val fg = if (solid) Bg else Fg
    Column(modifier.height(200.dp).clip(RoundedCornerShape(16.dp)).background(if (solid) Fg else Soft).clickable(role = Role.Button, onClick = onClick),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically)) {
        Icon(icon, contentDescription = null, Modifier.size(48.dp), tint = fg)
        Text(title, style = MaterialTheme.typography.headlineSmall, color = fg)
    }
}

@Composable
private fun Heading(title: String, trailing: @Composable () -> Unit = {}) =
    Row(Modifier.fillMaxWidth().heightIn(min = 40.dp).padding(bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(title, style = MaterialTheme.typography.titleLarge, color = Fg, modifier = Modifier.weight(1f))
        trailing()
    }

// ai: a list's columns: Name over the icon and the name, then the two figures' titles; the actions' column untitled
private val C2 = 200.dp
private val C3 = 220.dp
private val C4 = 160.dp

@Composable
private fun Columns(second: String, third: String) {
    val st = MaterialTheme.typography.labelSmall.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
    Row(Modifier.fillMaxWidth().padding(bottom = 8.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        Text("Name", style = st, color = Muted, modifier = Modifier.weight(1f))
        Text(second, style = st, color = Muted, modifier = Modifier.width(C2))
        Text(third, style = st, color = Muted, modifier = Modifier.width(C3))
        Spacer(Modifier.width(C4))
    }
    HorizontalDivider(color = Line)
}

@Composable
private fun ListLine(lead: ImageVector, name: String, onOpen: (() -> Unit)?, second: @Composable () -> Unit, third: String, actions: @Composable () -> Unit) {
    Row(Modifier.fillMaxWidth().heightIn(min = 60.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Box(Modifier.size(40.dp).clip(RoundedCornerShape(10.dp)).background(Soft), contentAlignment = Alignment.Center) {
                Icon(lead, contentDescription = null, Modifier.size(22.dp), tint = Muted)
            }
            Text(name, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = Fg, maxLines = 1, overflow = TextOverflow.Ellipsis,
                modifier = if (onOpen != null) Modifier.clickable(onClick = onOpen) else Modifier)
        }
        Box(Modifier.width(C2)) { second() }
        Text(third, style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1, modifier = Modifier.width(C3))
        Row(Modifier.width(C4).offset(x = 6.dp), horizontalArrangement = Arrangement.spacedBy(4.dp, Alignment.End), verticalAlignment = Alignment.CenterVertically) { actions() }
    }
    HorizontalDivider(color = Line)
}

@Composable
private fun PartialLine(h: HomeHost, p: Partial) {
    var asking by remember { mutableStateOf(false) }
    val frac = (h.liveFraction(p) ?: p.fraction).coerceIn(0.0, 1.0)
    val name = p.name.ifEmpty { "a file" }
    ListLine(Icons.file, name, { h.goReceive() }, {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Box(Modifier.width(64.dp).height(6.dp).clip(RoundedCornerShape(3.dp)).background(Soft)) {
                if (frac > 0) Box(Modifier.fillMaxWidth(frac.toFloat()).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(Fg))
            }
            Text("${floor(100 * frac).toInt()}% of ${Readout.bytes(p.length)}", style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1)
        }
    }, whenOf(p.at)) {
        Btn("Continue") { h.goReceive() }
        MoreMenu("More for $name", listOf(Act("Delete", danger = true) { asking = true }))
    }
    if (asking) AskDialog("Delete $name?", "Delete", { asking = false }) { h.deletePartial(p) }
}

@Composable
private fun FileLine(e: FileItem) {
    var asking by remember { mutableStateOf(false) }
    val size = listOf(if (e.bundle) countOf(e.files) else "", Readout.bytes(e.size)).filter { it.isNotEmpty() }.joinToString(", ")
    ListLine(if (e.bundle) Icons.folder else Icons.file, e.name, e.open, { Text(size, style = MaterialTheme.typography.bodyMedium, color = Muted, maxLines = 1) }, whenOf(e.at)) {
        Btn("Open") { e.open() }
        MoreMenu("More for ${e.name}", e.acts + Act("Delete", danger = true) { asking = true })
    }
    if (asking) AskDialog("Delete ${e.name}?", "Delete", { asking = false }) { e.delete() }
}
