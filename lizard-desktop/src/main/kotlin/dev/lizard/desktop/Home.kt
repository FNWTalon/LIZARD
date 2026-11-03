package dev.lizard.desktop

import dev.lizard.readout.Readout
import dev.lizard.ui.Act
import dev.lizard.ui.FileItem
import dev.lizard.ui.HomeHost
import dev.lizard.ui.Partial
import dev.lizard.ui.Screen

// ai: Home on the desktop (the screen is ../lizard-ui's HomeScreen): the receiver's files and unfinished transfers, a
// ai: file's Open and Show in folder; Send the solid tile, as the web's on a desktop
class DesktopHome(private val app: App) : HomeHost {
    private val r get() = app.recv
    override val sendSolid = true
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
    override val about: String get() = app.about
}
