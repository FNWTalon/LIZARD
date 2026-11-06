package dev.lizard.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog

// ai: A work LIZARD carries and its license, as the repository's NOTICE lists them (each package's license text holds
// ai: them in full), both named as About lists them
class Credit(val name: String, val license: String)

// ai: What both apps carry: the code compiled into the core, and the fonts. About lists the licenses in the order they
// ai: first come here, then in an app's own
val CREDITS = listOf(
    Credit("Vulkan headers", "Apache 2.0"),
    Credit("BLAKE3", "Apache 2.0 / CC0 1.0"),
    Credit("Wirehair", "BSD 3-Clause"),
    Credit("Zstandard", "BSD 3-Clause"),
    Credit("volk", "MIT"),
    Credit("JSON for Modern C++", "MIT"),
    Credit("Inter", "SIL OFL 1.1"),
    Credit("JetBrains Mono", "SIL OFL 1.1"),
)

// ai: what an app's own build carries besides (LizardTheme's `credits`: the desktop's runtime and the libraries its
// ai: packages bring, the phone's AndroidX)
val LocalCredits = staticCompositionLocalOf { emptyList<Credit>() }

// ai: The repository, opened in the platform's browser (LizardTheme's `browse`: Android's ACTION_VIEW, the desktop's
// ai: browser)
const val SOURCE_URL = "https://github.com/FNWTalon/LIZARD"
val LocalBrowse = staticCompositionLocalOf<(String) -> Unit> { {} }

// ai: Source code (2026-11-03; the web's #source): a row after About on Home, Send and Receive, drawn as a fold's row (a
// ai: hairline over it, 56 dp), opening the repository
@Composable
fun SourceRow() {
    val browse = LocalBrowse.current
    HorizontalDivider(color = Line)
    ListRow("Source code", onClick = { browse(SOURCE_URL) })
}

// ai: About's list, a line a license: "MIT: volk, JSON for Modern C++", the licenses in the order they first come, each
// ai: line's works in theirs
fun creditLines(cs: List<Credit>): List<String> {
    val groups = LinkedHashMap<String, MutableList<String>>()
    for (c in cs) groups.getOrPut(c.license) { mutableListOf() } += c.name
    return groups.map { (license, names) -> "$license: ${names.joinToString(", ")}" }
}

// ai: About (2026-11-02; a row with the build line before): a fold, closed until opened and not kept, its text in
// ai: paragraphs: the app's about line, LIZARD's license, the design it is based on, then a line a license of what both
// ai: apps and this one carry
@Composable
fun AboutFold(about: String) {
    var open by remember { mutableStateOf(false) }
    Fold("About", open, { open = !open }) { AboutText(about) }
}

// ai: About's paragraphs, the fold's and the PC Home's dialog's (2026-11-06)
@Composable
fun AboutText(about: String) {
    val body = MaterialTheme.typography.bodyMedium
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        if (about.isNotEmpty()) Text(about, style = body, color = Muted)
        Text("LIZARD is licensed under the Mozilla Public License 2.0 (MPL 2.0).", style = body, color = Fg)
        Text("The design is based on FOCUS: Robust Visual Codes for Everyone (Hermans et al., MobiSys 2016). The " +
            "implementation is independent.", style = body, color = Fg)
        Text("Third-party software and licenses:", style = body, color = Fg)
        for (line in creditLines(CREDITS + LocalCredits.current)) Text(line, style = body, color = Fg)
    }
}

// ai: About as a dialog (the PC Home's bar, 2026-11-06; the web's #aboutDlg): the paragraphs and Close, on white, 12 dp corners
@Composable
fun AboutDialog(about: String, onDismiss: () -> Unit) =
    AlertDialog(onDismissRequest = onDismiss, confirmButton = { Btn("Close", onClick = onDismiss) },
        title = { Text("About", style = MaterialTheme.typography.titleMedium, color = Fg) }, text = { AboutText(about) },
        shape = RoundedCornerShape(12.dp), containerColor = Bg)
