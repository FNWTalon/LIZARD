package dev.lizard.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
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

// ai: A work LIZARD carries and its license, as the repository's NOTICE lists them (each package's license text holds
// ai: them in full); the name as a sentence reads it
class Credit(val name: String, val license: String)

// ai: What both apps carry: the code compiled into the core, and the fonts
val CREDITS = listOf(
    Credit("BLAKE3", "Apache License 2.0 or CC0 1.0"),
    Credit("Wirehair", "BSD 3-Clause"),
    Credit("Zstandard", "BSD 3-Clause"),
    Credit("volk", "MIT"),
    Credit("JSON for Modern C++", "MIT"),
    Credit("the Vulkan headers", "Apache License 2.0"),
    Credit("the Inter and JetBrains Mono fonts", "SIL Open Font License 1.1"),
)

// ai: what an app's own build carries besides (LizardTheme's `credits`: the desktop's runtime and the libraries its
// ai: packages bring, the phone's AndroidX)
val LocalCredits = staticCompositionLocalOf { emptyList<Credit>() }

// ai: the credits as one run of a sentence: the works of one license together, in the order they first come, each group
// ai: its license in brackets, the groups apart by semicolons ("Wirehair and Zstandard (BSD 3-Clause); volk and ...")
fun creditsLine(cs: List<Credit>): String {
    val groups = LinkedHashMap<String, MutableList<String>>()
    for (c in cs) groups.getOrPut(c.license) { mutableListOf() } += c.name
    fun and(xs: List<String>) = if (xs.size < 2) xs.joinToString() else xs.dropLast(1).joinToString(", ") + " and " + xs.last()
    val parts = groups.map { (license, names) -> "${and(names)} ($license)" }
    return if (parts.size < 2) parts.joinToString() else parts.dropLast(1).joinToString("; ") + "; and " + parts.last()
}

// ai: About (2026-11-02; a row with the build line before): a fold, closed until opened and not kept, its text in
// ai: paragraphs: the app's about line, LIZARD's license and the design it derives from, then what it includes and what
// ai: this app carries besides
@Composable
fun AboutFold(about: String) {
    var open by remember { mutableStateOf(false) }
    Fold("About", open, { open = !open }) {
        val body = MaterialTheme.typography.bodyMedium
        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (about.isNotEmpty()) Text(about, style = body, color = Muted)
            Text("LIZARD is licensed under the Mozilla Public License 2.0. Its design derives from FOCUS: Robust Visual Codes " +
                "for Everyone, by F. Hermans, L. McNamara, G. Sörös, C. Rohner, T. Voigt and E. Ngai (MobiSys 2016); the code is " +
                "its own.", style = body, color = Fg)
            Text("It includes ${creditsLine(CREDITS)}.", style = body, color = Fg)
            val own = LocalCredits.current
            if (own.isNotEmpty()) Text("This app also carries ${creditsLine(own)}.", style = body, color = Fg)
        }
    }
}
