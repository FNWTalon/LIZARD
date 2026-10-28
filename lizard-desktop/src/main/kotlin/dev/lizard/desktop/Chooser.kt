package dev.lizard.desktop

import java.awt.Component
import java.awt.EventQueue
import java.awt.FileDialog
import java.awt.Frame
import java.io.File
import javax.swing.JFileChooser
import javax.swing.SwingUtilities
import kotlin.concurrent.thread

// ai: The chooser (2026-10-23 in SendState, Swing's folder chooser before, which looks like no desktop's; its own file
// ai: since 2026-10-28, for Receive's Upload too): the desktop's own dialog through the XDG desktop portal
// ai: (Native.portalPick: GTK's on Cinnamon and GNOME, KDE's on Plasma; Windows' shell dialog), attached to the window
// ai: the owner is in, on a thread of its own while it is open; where no portal answers, zenity's (GTK's too); else
// ai: AWT's FileDialog for files (GTK's; the system's open panel on macOS, folders too there) and Swing's
// ai: JFileChooser for folders, which AWT's cannot pick. One open at a time.
internal object Chooser {
    private var open = false   // ai: the UI thread's

    // ai: then: on the UI thread with what was chosen; not called where nothing was
    fun choose(owner: Component, title: String, folders: Boolean, multiple: Boolean, then: (List<File>) -> Unit) {
        if (open) return
        open = true
        val parent = runCatching { Native.windowOf(owner) }.getOrDefault(0L)
        thread(name = "lizard-chooser", isDaemon = true) {
            val got = portal(parent, title, folders, multiple) ?: zenity(title, folders, multiple)
            EventQueue.invokeLater {
                open = false
                val files = got ?: fallback(owner, title, folders, multiple)
                if (files.isNotEmpty()) then(files)
            }
        }
    }

    // ai: the portal's choice as files (its file:// URIs; Windows' dialog gives paths), none when cancelled, null where it
    // ai: could not be asked
    private fun portal(parent: Long, title: String, folders: Boolean, multiple: Boolean): List<File>? {
        if (Native.mac) return null
        val uris = runCatching { Native.portalPick(parent, title, folders, multiple) }.getOrNull()
        if (uris == null) { System.err.println("chooser: no portal: ${runCatching { Native.txError() }.getOrDefault("")}"); return null }
        return uris.mapNotNull { u -> runCatching { if (u.startsWith("file:")) File(java.net.URI(u)) else File(u) }.getOrNull() }
    }
    // ai: zenity's, a path a line; none when cancelled (exit 1), null where it is not there
    private fun zenity(title: String, folders: Boolean, multiple: Boolean): List<File>? = if (Native.mac || Native.windows) null else runCatching {
        val p = ProcessBuilder(listOf("zenity", "--file-selection", "--separator=\n", "--title=$title") +
            (if (multiple) listOf("--multiple") else emptyList()) + (if (folders) listOf("--directory") else emptyList()))
            .redirectError(ProcessBuilder.Redirect.DISCARD).start()
        val out = p.inputStream.bufferedReader().readText()
        when (p.waitFor()) { 0 -> out.lines().filter { it.isNotEmpty() }.map(::File); 1 -> emptyList(); else -> null }
    }.getOrNull()
    // ai: the dialogs before the portal, over the owner's window
    private fun fallback(owner: Component, title: String, folders: Boolean, multiple: Boolean): List<File> {
        val frame = owner as? Frame ?: SwingUtilities.getWindowAncestor(owner) as? Frame
        if (Native.mac || !folders) {
            if (Native.mac) System.setProperty("apple.awt.fileDialogForDirectories", folders.toString())
            val d = FileDialog(frame, title, FileDialog.LOAD)
            d.isMultipleMode = multiple
            d.isVisible = true
            if (Native.mac) System.setProperty("apple.awt.fileDialogForDirectories", "false")
            return d.files.toList()
        }
        val c = JFileChooser().apply { fileSelectionMode = JFileChooser.DIRECTORIES_ONLY; dialogTitle = "Choose a folder" }
        return if (c.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) listOfNotNull(c.selectedFile) else emptyList()
    }
}
