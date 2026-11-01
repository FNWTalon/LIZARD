package dev.lizard.receiver

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import dev.lizard.readout.Readout
import dev.lizard.ui.Act
import dev.lizard.ui.BundleFile
import dev.lizard.ui.FileItem
import dev.lizard.ui.HomeHost
import dev.lizard.ui.Partial
import dev.lizard.ui.Screen

// ai: Home on the phone (the screen is ../lizard-ui's HomeScreen; its bundle screen BundleScreen): the received files in
// ai: Downloads (Received.kt: a press opens a file, or a bundle's own screen; the menu Open, Share, Save a copy or Save to
// ai: a folder), the unfinished transfers, the phone's words, the tips kept in the preferences; Receive the solid tile,
// ai: as the web's on a touch screen
internal class AndroidHome(private val a: MainActivity) : HomeHost {
    override val tipsSeen get() = a.tipsSeen
    override fun tipsDone() = a.tipsDone()
    override fun tipsAgain() = a.tipsAgain()
    override val tipSteps = listOf("On the sending device, open LIZARD, choose Send and pick a file.", "On the receiving phone, open LIZARD and choose Receive.",
        "Point it at the code until the file arrives. It is kept here.")
    // ai: the example on video (the README's), a link the system opens
    override fun openExample() { try { a.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(EXAMPLE_URL))) } catch (_: ActivityNotFoundException) {} }
    override val sendSolid = false
    override val receiveSub = "Read a code with this phone's camera"
    override fun goSend() = a.go(Screen.Send)
    override fun goReceive() = a.go(Screen.Receive)
    override fun refresh() {}
    override val partials: List<Partial> get() = a.partials
    override fun liveFraction(p: Partial): Double? = if (p.root.isNotEmpty() && p.root == a.root && a.rx.hasFile) Readout.fraction(a.rx) else null
    override fun deletePartial(p: Partial) = a.deletePartial(p)
    override val files: List<FileItem> get() = a.files.map { e ->
        FileItem(e.id, e.name, e.bundle, e.files, e.size, e.at, open = { a.open(e) },
            acts = listOf(Act("Open") { a.open(e) }, Act("Share") { a.share(e) }, Act(if (e.bundle) "Save to a folder" else "Save a copy") { a.saveCopy(e) }),
            delete = { a.delete(e) })
    }
    override fun deleteAll() = a.deleteAll()
    override val note get() = a.note
    override val device = "phone"
    override val about get() = a.aboutLine()

    // ai: the bundle a press opened (MainActivity.shownBundle): its files sorted by their path in the folder
    val bundleActs get() = a.shownBundle?.let { e -> listOf(Act("Share") { a.share(e) }, Act("Save") { a.saveCopy(e) }) }.orEmpty()
    val bundleFiles get() = a.shownBundle?.items?.sortedBy { it.path }?.map { it ->
        val mime = Received.mimeOf(it.path)
        BundleFile(it.path, it.size, open = { a.openUri(a.received.uriOf(it.uri), mime) },
            acts = listOf(Act("Open") { a.openUri(a.received.uriOf(it.uri), mime) }, Act("Share") { a.shareUri(a.received.uriOf(it.uri), it.path.substringAfterLast('/'), mime) }))
    }.orEmpty()

    companion object {
        private const val EXAMPLE_URL = "https://www.youtube.com/shorts/pjcAtQ7GcVQ"
    }
}
