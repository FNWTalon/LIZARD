package dev.lizard.desktop

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import dev.lizard.readout.Readout
import dev.lizard.ui.Act
import dev.lizard.ui.FileItem
import dev.lizard.ui.HomeHost
import dev.lizard.ui.Partial
import dev.lizard.ui.Screen

// ai: Home on the desktop (the screen is ../lizard-ui's HomeScreen): the receiver's files and unfinished transfers, the
// ai: computer's words, a file's Open and Show in folder, the tips kept in Prefs; Send the solid tile, as the web's on a
// ai: desktop
class DesktopHome(private val app: App) : HomeHost {
    private val r get() = app.recv
    private var seen by mutableStateOf(Prefs.Rx.tipsSeen)
    override val tipsSeen get() = seen
    override fun tipsDone() { seen = true; Prefs.Rx.tipsSeen = true }
    override fun tipsAgain() { seen = false; Prefs.Rx.tipsSeen = false }
    override val tipSteps = listOf("On the computer, open LIZARD, choose Send and pick a file.", "On the phone, open LIZARD and choose Receive.",
        "Point the phone at the code until the file arrives. It is kept here.")
    override fun openExample() { Received.browse(EXAMPLE_URL) }
    override val sendSolid = true
    override val receiveSub = "Read a code with this device's camera"
    override fun goSend() = app.go(Screen.Send)
    override fun goReceive() = app.go(Screen.Receive)
    override fun refresh() = r.refreshLists()
    override val partials: List<Partial> get() = r.partials
    override fun liveFraction(p: Partial): Double? = if (p.root.isNotEmpty() && p.root == r.root && r.rx.hasFile) Readout.fraction(r.rx) else null
    override fun deletePartial(p: Partial) = r.deletePartial(p)
    override val files: List<FileItem> get() = r.files.map { e ->
        FileItem(e.path, e.name, e.bundle, e.files, e.size, e.at, open = { Received.open(e.file) },
            acts = listOf(Act("Open") { Received.open(e.file) }, Act("Show in folder") { Received.show(e.file) }), delete = { r.deleteReceived(e) })
    }
    override fun deleteAll() = r.deleteAll()
    override val note = ""
    override val device = "computer"
    override val about: String get() = app.about

    companion object {
        // ai: the example on video, the README's
        private const val EXAMPLE_URL = "https://www.youtube.com/shorts/pjcAtQ7GcVQ"
    }
}
