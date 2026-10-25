package dev.lizard.desktop

import androidx.compose.foundation.Image
import androidx.compose.foundation.VerticalScrollbar
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
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.lizard.readout.Readout
import java.text.DateFormat
import java.util.Date
import kotlin.math.floor

// ai: Home (2026-10-25 on the desktop; the web's index.html, the Android app's Home.kt): the bar with the mark and the
// ai: name, the first-run tip until it is dismissed, Send and Receive side by side (Send the solid one, as the web's
// ai: on a desktop), the unfinished transfers (shown where there are any: a click or Continue opens Receive with the
// ai: camera, which resumes one once its file is in the light), the received files (a click opens one; its menu Open,
// ai: Show in folder, Delete) or the empty state, and Settings (the tips again, the received files with Delete all,
// ai: About). One column, at most the web's 34rem, centred; the bar fixed, the rest scrolling under it.
@Composable
fun HomeScreen(app: App) {
    val r = app.recv
    // ai: the lists again on entering, and the unfinished transfers a second (a transfer in hand writes its meta.json as
    // ai: it goes; ReceiveState's poll reads them while a session lives)
    LaunchedEffect(Unit) { r.refreshLists() }
    Column(Modifier.fillMaxSize().background(Bg)) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.TopCenter) {
            Box(Modifier.widthIn(max = 544.dp).fillMaxWidth().padding(horizontal = 16.dp)) { HomeBar() }
        }
        Box(Modifier.weight(1f).fillMaxWidth()) {
            val scroll = rememberScrollState()
            Box(Modifier.fillMaxSize().verticalScroll(scroll), contentAlignment = Alignment.TopCenter) {
                Column(Modifier.widthIn(max = 544.dp).fillMaxWidth().padding(horizontal = 16.dp)) {
                    HomeColumn(app)
                    Spacer(Modifier.height(32.dp))
                }
            }
            VerticalScrollbar(rememberScrollbarAdapter(scroll), Modifier.align(Alignment.CenterEnd).fillMaxHeight())
        }
    }
}

// ai: The bar (the web's .bar .brand): the mark at 28 dp and LIZARD at 20 sp bold, a hairline under it
@Composable
private fun HomeBar() = Column {
    Row(Modifier.fillMaxWidth().height(56.dp), verticalAlignment = Alignment.CenterVertically) {
        Image(rememberVectorPainter(Icons.mark), contentDescription = null, Modifier.size(28.dp))
        Spacer(Modifier.width(10.dp))
        Text("LIZARD", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = Fg)
    }
    HorizontalDivider(color = Line)
}

// ai: the example on video, the README's
private const val EXAMPLE_URL = "https://www.youtube.com/shorts/pjcAtQ7GcVQ"

private fun whenOf(at: Long) = if (at > 0) DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(at)) else ""
private fun count(n: Int) = "$n ${if (n == 1) "file" else "files"}"

@Composable
private fun ColumnScope.HomeColumn(app: App) {
    val r = app.recv
    var tips by remember { mutableStateOf(!Prefs.Rx.tipsSeen) }
    if (tips) TipCard(onGot = { tips = false; Prefs.Rx.tipsSeen = true })
    Row(Modifier.fillMaxWidth().padding(top = 16.dp, bottom = 8.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Tile(Icons.monitor, "Send", "Show a file as a code on this screen", solid = true, modifier = Modifier.weight(1f)) { app.go(Screen.Send) }
        Tile(Icons.camera, "Receive", "Read a code with this device's camera", solid = false, modifier = Modifier.weight(1f)) { app.go(Screen.Receive) }
    }
    if (r.partials.isNotEmpty()) {
        SectionLabel("Unfinished")
        r.partials.forEachIndexed { i, p -> key(p.id) { PartialRow(app, p, last = i == r.partials.lastIndex) } }
    }
    SectionLabel("Received files")
    if (r.files.isEmpty()) Column(Modifier.fillMaxWidth().padding(vertical = 24.dp), horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Icon(Icons.file, contentDescription = null, Modifier.size(24.dp), tint = Muted)
        Text("Files you receive appear here.", style = MaterialTheme.typography.bodyMedium, color = Muted)
    }
    // ai: keyed by the file, so a row's menu and its question stay with it when a file is filed above it meanwhile
    r.files.forEachIndexed { i, e -> key(e.path) { FileRow(r, e, last = i == r.files.lastIndex) } }
    HomeSettings(app) { tips = true; Prefs.Rx.tipsSeen = false }
}

// ai: The first-run tip (the web's .tip): the info icon, the one line in bold, the three steps, the example on video, a
// ai: text "Got it"; on the card fill
@Composable
private fun TipCard(onGot: () -> Unit) {
    Row(Modifier.fillMaxWidth().padding(top = 12.dp).clip(RoundedCornerShape(12.dp)).background(Card)
        .padding(start = 16.dp, end = 16.dp, top = 14.dp, bottom = 6.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Icon(Icons.info, contentDescription = null, Modifier.size(24.dp), tint = Muted)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Send a file from one screen to another device's camera.", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.SemiBold, color = Fg)
            Column {
                for ((i, step) in listOf("On the computer, open LIZARD, choose Send and pick a file.", "On the phone, open LIZARD and choose Receive.",
                    "Point the phone at the code until the file arrives. It is kept here.").withIndex())
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        // ai: the numbers in a column of their own, so every step's text starts at one edge (the web's ol)
                        Text("${i + 1}.", style = MaterialTheme.typography.bodyLarge, color = Fg, modifier = Modifier.width(18.dp))
                        Text(step, style = MaterialTheme.typography.bodyLarge, color = Fg)
                    }
            }
            Text("An example on video", style = MaterialTheme.typography.bodyLarge.copy(textDecoration = TextDecoration.Underline), color = Fg,
                modifier = Modifier.pointerHoverIcon(PointerIcon.Hand).clickable(role = Role.Button) { Received.browse(EXAMPLE_URL) })
            Btn("Got it", Kind.Text, modifier = Modifier.offset(x = (-12).dp), onClick = onGot)
        }
    }
}

// ai: A card of Home's (the web's .go .btn): an icon, its title, a line under it; the whole card the action; the solid
// ai: one the foreground's fill, its line in white at 72%
@Composable
private fun Tile(icon: ImageVector, title: String, sub: String, solid: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val fg = if (solid) Bg else Fg
    Column(modifier.heightIn(min = 120.dp).clip(RoundedCornerShape(16.dp)).background(if (solid) Fg else Soft)
        .clickable(role = Role.Button, onClick = onClick).padding(16.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Icon(icon, contentDescription = null, Modifier.size(24.dp), tint = fg)
        Spacer(Modifier.height(8.dp))
        Text(title, style = MaterialTheme.typography.titleMedium, fontSize = 18.sp, color = fg)
        Text(sub, style = MaterialTheme.typography.bodyMedium, color = if (solid) Bg.copy(alpha = .72f) else Muted)
    }
}

// ai: An unfinished transfer's row: its name (or "a file") and "37% of 3.1 MB · date" (the receiver's live share where
// ai: it holds that root); a click opens Receive with the camera, its menu Continue (the same) and Delete, asked first
@Composable
private fun PartialRow(app: App, p: Partials.Entry, last: Boolean) {
    val r = app.recv
    var asking by remember { mutableStateOf(false) }
    val frac = if (p.root.isNotEmpty() && p.root == r.root && r.rx.hasFile) Readout.fraction(r.rx) else p.fraction
    val name = p.name.ifEmpty { "a file" }
    ListRow(name, listOf("${floor(100 * frac).toInt()}% of ${Readout.bytes(p.length)}", whenOf(p.at)).filter { it.isNotEmpty() }.joinToString(" · "),
        lead = Icons.file, divider = !last, onClick = { app.go(Screen.Receive) }) {
        MoreMenu("More for $name", listOf(Act("Continue") { app.go(Screen.Receive) }, Act("Delete", danger = true) { asking = true }))
    }
    if (asking) AskDialog("Delete $name?", "The part received goes from this computer.", "Delete", { asking = false }) { r.deletePartial(p) }
}

// ai: A received file's row, or a bundle's (several files or a folder sent as one, unpacked: a folder, its files'
// ai: count before the size): a click opens it; its menu Open, Show in folder, Delete, asked first
@Composable
private fun FileRow(r: ReceiveState, e: Received.Entry, last: Boolean) {
    var asking by remember { mutableStateOf(false) }
    ListRow(e.name, listOf(if (e.bundle) count(e.files) else "", Readout.bytes(e.size), whenOf(e.at)).filter { it.isNotEmpty() }.joinToString(" · "),
        lead = if (e.bundle) Icons.folder else Icons.file, divider = !last, onClick = { Received.open(e.file) }) {
        MoreMenu("More for ${e.name}", listOf(Act("Open") { Received.open(e.file) }, Act("Show in folder") { Received.show(e.file) },
            Act("Delete", danger = true) { asking = true }))
    }
    if (asking) AskDialog("Delete ${e.name}?", "It goes from this computer.", "Delete", { asking = false }) { r.deleteReceived(e) }
}

// ai: Home's settings (the web Home's .settings): the tips again, the received files' count and size with Delete all
// ai: (asked first; the unfinished transfers stay), About
@Composable
private fun HomeSettings(app: App, tipsAgain: () -> Unit) {
    val r = app.recv
    var asking by remember { mutableStateOf(false) }
    SectionLabel("Settings")
    ListRow("Show tips again", "The first-run tips, on every page", divider = true, onClick = tipsAgain)
    val n = r.files.size
    ListRow("Received files", if (n == 0) "No files kept." else "$n ${if (n == 1) "file" else "files"} kept, ${Readout.bytes(r.files.sumOf { it.size })}", divider = true) {
        if (n > 0) Btn("Delete all", Kind.Danger) { asking = true }
    }
    ListRow("About", app.about.ifEmpty { null })
    if (asking) AskDialog("Delete $n ${if (n == 1) "file" else "files"}?", "They go from this computer.", "Delete all", { asking = false }) { r.deleteAll() }
}
