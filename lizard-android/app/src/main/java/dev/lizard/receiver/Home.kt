package dev.lizard.receiver

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withLink
import androidx.compose.ui.unit.dp
import java.text.DateFormat
import java.util.Date

// ai: Home (2026-10-01): the first-run tip (the web Home's three steps, worded for the phone), Send and Receive (Receive
// ai: the one solid card; the app sends since the same day, Send.kt), the received files (Library.kt: a row opens its file; its more
// ai: menu: Open, Share, Save a copy, Delete, asked first) or the empty state, and Settings at the bottom (the web Home's;
// ai: a gear to a Settings screen until 2026-10-02). One
// ai: column, at most 640 dp wide, the same in either orientation, scrolling under the bar.
@Composable
internal fun MainActivity.HomeScreen() {
    Page(null, null) {
        Spacer(Modifier.height(8.dp))
        if (!tipsSeen) TipCard(onGot = { tipsDone() }) {
            Text("Send a file from one screen to another device's camera.", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.SemiBold, color = Fg)
            Steps("On the sending device, open Lizard, choose Send and pick a file.", "On the receiving phone, open Lizard and choose Receive.",
                "Point it at the code until the file arrives. It is kept here.")
            // ai: the example on video (the README's), a link the system opens
            Text(buildAnnotatedString {
                withLink(LinkAnnotation.Url(EXAMPLE_URL, TextLinkStyles(SpanStyle(textDecoration = TextDecoration.Underline)))) { append("An example on video") }
            }, style = MaterialTheme.typography.bodyLarge, color = Fg, modifier = Modifier.padding(top = 4.dp))
        }
        // ai: Send and Receive side by side, as the web Home's tiles; Receive the solid one, as the web's on a touch screen
        Row(Modifier.fillMaxWidth().padding(vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Tile(R.drawable.ic_send, "Send", "Show a file as a code on this screen", solid = false, modifier = Modifier.weight(1f)) { go(MainActivity.Screen.Send) }
            Tile(R.drawable.ic_camera, "Receive", "Read a code with this phone's camera", solid = true, modifier = Modifier.weight(1f)) { go(MainActivity.Screen.Receive) }
        }
        SectionLabel("Received files")
        if (files.isEmpty()) Column(Modifier.fillMaxWidth().padding(vertical = 24.dp), horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(painterResource(R.drawable.ic_file), contentDescription = null, Modifier.size(24.dp), tint = Muted)
            Text("Files you receive appear here.", style = MaterialTheme.typography.bodyMedium, color = Muted)
        }
        // ai: keyed by the file, so a row's menu and its Delete question stay with it when a file is kept above it
        // ai: meanwhile (the stats poll files on any screen; 2026-10-01: they followed the position)
        files.forEachIndexed { i, e -> key(e.id) { FileRow(e, last = i == files.lastIndex) } }
        if (note.isNotEmpty()) Text(note, style = MaterialTheme.typography.bodyMedium, color = if (note.startsWith("Saved")) Muted else Bad, modifier = Modifier.padding(top = 8.dp))
        HomeSettings()
    }
}

// ai: A card of Home's (the web's .go .btn): an icon, its title, a line under it; the whole card the action
@Composable
private fun Tile(icon: Int, title: String, sub: String, solid: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val fg = if (solid) Bg else Fg
    Column(modifier.heightIn(min = 120.dp).clip(RoundedCornerShape(16.dp)).background(if (solid) Fg else Soft)
        .clickable(role = Role.Button, onClick = onClick).padding(16.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Icon(painterResource(icon), contentDescription = null, Modifier.size(24.dp), tint = fg)
        Spacer(Modifier.height(8.dp))
        Text(title, style = MaterialTheme.typography.titleMedium, color = fg)
        Text(sub, style = MaterialTheme.typography.bodyMedium, color = if (solid) Bg.copy(alpha = .72f) else Muted)
    }
}

// ai: The tip's numbered steps (the web's ol)
@Composable
internal fun Steps(vararg steps: String) {
    steps.forEachIndexed { i, s ->
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("${i + 1}.", style = MaterialTheme.typography.bodyLarge, color = Fg)
            Text(s, style = MaterialTheme.typography.bodyLarge, color = Fg)
        }
    }
}

// ai: the example on video, the README's
private const val EXAMPLE_URL = "https://www.youtube.com/watch?v=F-Mie4m9gBQ"

private fun whenOf(at: Long) = if (at > 0) DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(at)) else ""

@Composable
internal fun MainActivity.FileRow(e: Library.Entry, last: Boolean) {
    var menu by remember { mutableStateOf(false) }
    var asking by remember { mutableStateOf(false) }
    ListRow(e.name, listOf(Readout.bytes(e.size), whenOf(e.at)).filter { it.isNotEmpty() }.joinToString(" · "), lead = R.drawable.ic_file,
        divider = !last, onClick = { open(e) }) {
        Box {
            IconBtn(R.drawable.ic_more, "More for ${e.name}") { menu = true }
            DropdownMenu(menu, { menu = false }) {
                DropdownMenuItem({ Text("Open") }, { menu = false; open(e) })
                DropdownMenuItem({ Text("Share") }, { menu = false; share(e) })
                DropdownMenuItem({ Text("Save a copy") }, { menu = false; saveCopy(e) })
                DropdownMenuItem({ Text("Delete", color = Bad) }, { menu = false; asking = true })
            }
        }
    }
    if (asking) AlertDialog(onDismissRequest = { asking = false },
        confirmButton = { Btn("Delete", Kind.Danger) { asking = false; delete(e) } },
        dismissButton = { Btn("Cancel", Kind.Text) { asking = false } },
        title = { Text("Delete ${e.name}?") },
        text = { Text("It goes from this phone. A copy saved elsewhere stays.") },
        containerColor = Bg)
}
