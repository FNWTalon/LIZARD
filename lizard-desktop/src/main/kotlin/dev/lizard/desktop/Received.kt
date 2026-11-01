package dev.lizard.desktop

import dev.lizard.Session
import dev.lizard.readout.Readout
import dev.lizard.tar.TAR_TYPE
import dev.lizard.tar.untar
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.Json
import java.awt.Desktop
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.channels.FileChannel
import java.nio.channels.FileLock
import java.nio.file.AccessDeniedException
import java.nio.file.FileSystemException
import java.nio.file.Files
import java.nio.file.NoSuchFileException
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.util.Locale
import kotlin.concurrent.thread
import dev.lizard.ui.*

// ai: The received files (2026-10-25, the PC receiver; the Android app's Library.kt, the web's library.mjs): a verified
// ai: file is taken out of the receiver's store by the receive session into the data folder's kept/ (liblizard/core/rx/
// ai: session.h: kept/<id>/ with filing.json and file/<name>, which no store's open empties; Session.kept lists them),
// ai: then filed from there into the Downloads folder and its take let go (Session.placed): file, into LIZ_DOWNLOADS,
// ai: else Linux's XDG_DOWNLOAD_DIR from user-dirs.dirs, else ~/Downloads, under a name free there ("name (1).ext"),
// ai: each file marked as downloaded
// ai: (mark); a tar (several files or a folder sent as one, Tar.kt) is unpacked there whole into a folder (one folder
// ai: alone at its top is the bundle's, else the tar's name less .tar), one that does not unpack whole filed as the file.
// ai: What Downloads does not take stays in the app's folder, listed from there, and is filed again later (retry), as is
// ai: a filing the app's close cut short: a file is never only in a store that lets it go. A take is filed when its file
// ai: is in Downloads, durable, and listed: its row is written under the list's lock just before the move into place
// ai: (taken out again where the move fails), and its take let go after, so a filing cut short leaves either the take
// ai: (filed again) or its row (the take let go at the next pass). The list the app keeps is Native.data/received.json
// ai: (entries name, path, type, size, root, at, bundle, files), newest first, written durably whole; an entry whose path
// ai: is gone is dropped from it, a list that does not read is never written over (one that does not parse is set aside
// ai: whole as received.json.bad), and the same root filed again replaces its entry. Every read and write of the list
// ai: is under one lock, from any thread. Open, Show in folder and Delete hand the file to the desktop: Delete to its
// ai: trash where java.awt.Desktop has one (Windows, macOS), else the file is deleted.
object Received {
    class Entry(val name: String, val path: String, val type: String, val size: Long, val root: String, val at: Long,
                val bundle: Boolean, val files: Int) {
        val file get() = File(path)
    }

    private val mu = Any()
    private val json = Json { ignoreUnknownKeys = true }
    private val listFile get() = File(Native.data, "received.json")
    private val home get() = System.getProperty("user.home")
    // ai: the session's takes (session.h): a folder each, filing.json (name, type, root) and file/<the file>
    private val keptDir get() = File(Native.data, "kept")

    // ai: where received files go: LIZ_DOWNLOADS (the tests'), else the desktop's own Downloads
    val downloads: File get() = System.getenv("LIZ_DOWNLOADS")?.takeIf { it.isNotEmpty() }?.let { File(it) } ?: xdgDownloads() ?: File(home, "Downloads")

    // ai: Linux: XDG_DOWNLOAD_DIR as user-dirs.dirs states it ("$HOME/..." or an absolute path), the folder a
    // ai: desktop names in its own language
    private fun xdgDownloads(): File? {
        if (Native.mac || Native.windows) return null
        val cfg = System.getenv("XDG_CONFIG_HOME")?.takeIf { it.isNotEmpty() } ?: "$home/.config"
        val line = runCatching { File(cfg, "user-dirs.dirs").readLines() }.getOrNull()
            ?.firstOrNull { it.trimStart().startsWith("XDG_DOWNLOAD_DIR=") } ?: return null
        val v = line.substringAfter('=').trim().removeSurrounding("\"").replace("\$HOME", home)
        return if (v.startsWith("/") && v.trimEnd('/') != home) File(v) else null
    }

    // ai: the entries whose files are there, newest first; a list that does not read gives none and is left as it is
    fun list(): List<Entry> = synchronized(mu) {
        val all = runCatching { read() }.getOrElse { System.err.println("received: ${listFile.path}: ${it.message}"); return emptyList() }
        val here = all.filter { File(it.path).exists() }
        if (here.size != all.size) runCatching { write(here) }.onFailure { System.err.println("received: ${listFile.path}: ${it.message}") }
        here.sortedByDescending { it.at }
    }

    // ai: the list as written: none where there is no list yet; a read that fails throws (nothing is written over the list
    // ai: then); one that does not parse is set aside whole beside it as received.json.bad, said in the log, and a new
    // ai: list begun
    private fun read(): List<Entry> {
        val f = listFile
        if (!f.exists()) return emptyList()
        val text = f.readText()
        val a = try { json.parseToJsonElement(text).jsonArray } catch (x: Exception) {
            val bad = File(f.parentFile, "received.json.bad")
            Files.move(f.toPath(), bad.toPath(), StandardCopyOption.REPLACE_EXISTING)
            System.err.println("received: ${f.path} did not parse (${text.length} B): set aside as ${bad.path}")
            return emptyList()
        }
        return a.mapNotNull { e ->
            val o = e as? JsonObject ?: return@mapNotNull null
            Entry(o.str("name"), o.str("path"), o.str("type"), o.num("size").toLong(), o.str("root"), o.num("at").toLong(), o.flag("bundle"), o.num("files").toInt())
                .takeIf { it.path.isNotEmpty() }
        }
    }

    // ai: written whole to a file beside it, made durable, then renamed over it, so a crash or a power loss leaves the
    // ai: list as it was
    private fun write(es: List<Entry>) {
        val f = listFile
        f.parentFile?.mkdirs()
        val tmp = File(f.parentFile, f.name + ".tmp")
        FileOutputStream(tmp).use { o -> o.write(JsonArray(es.map { asJson(it) }).toString().toByteArray()); o.fd.sync() }
        Files.move(tmp.toPath(), f.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    }

    fun asJson(e: Entry) = JsonObject(mapOf("name" to JsonPrimitive(e.name), "path" to JsonPrimitive(e.path), "type" to JsonPrimitive(e.type),
        "size" to JsonPrimitive(e.size), "root" to JsonPrimitive(e.root), "at" to JsonPrimitive(e.at), "bundle" to JsonPrimitive(e.bundle),
        "files" to JsonPrimitive(e.files)))

    private fun add(e: Entry): Entry = synchronized(mu) {
        write(read().filter { it.root != e.root || e.root.isEmpty() } + e)
        e
    }
    // ai: from into Downloads at dest, listed (e, whose path is dest) in one step under the list's lock: the row first,
    // ai: then the move (no list() between them drops the row of a file not yet there), the row taken out again where the
    // ai: move fails; whether it moved
    private fun moveListed(from: File, dest: File, e: Entry): Boolean = synchronized(mu) {
        add(e)
        if (from.renameTo(dest)) return true
        write(read().filter { it.path != e.path })
        false
    }

    // ai: a name not taken in dir: name, else "stem (1).ext", "stem (2).ext", ... (a folder's: "name (1)")
    fun free(dir: File, name: String): File {
        val f = File(dir, name)
        if (!f.exists()) return f
        val dot = name.lastIndexOf('.')
        val (stem, ext) = if (dot > 0) name.substring(0, dot) to name.substring(dot) else name to ""
        var k = 1
        while (true) { val g = File(dir, "$stem ($k)$ext"); if (!g.exists()) return g; k++ }
    }

    // ai: a take as the session lists it (Session.kept): its folder's id, the file's name, type and root, and the file
    private class Kept(val id: String, val name: String, val type: String, val root: String, val file: File)
    private fun kept(): List<Kept> = runCatching {
        json.parseToJsonElement(Session.kept(Native.data.path)).jsonArray.mapNotNull { e ->
            val o = e.jsonObject
            Kept(o.str("id"), o.str("name"), o.str("type"), o.str("root"), File(o.str("path"))).takeIf { it.id.isNotEmpty() && it.name.isNotEmpty() }
        }
    }.getOrDefault(emptyList())
    // ai: the take let go once its file is placed (or deleted with its entry)
    private fun placed(id: String) {
        val why = Session.placed(Native.data.path, id)
        if (why.isNotEmpty()) System.err.println("received: kept/$id: $why")
    }

    // ai: What a filing came to: the take's root and name, and "" where the file is in Downloads, else why Downloads did
    // ai: not take it (the file is in the app's folder then, listed from there)
    class Filed(val root: String, val name: String, val why: String)

    // ai: A take filed (the caller's thread, one filing at a time: ReceiveState's files' thread): a root a list entry
    // ai: holds with its file there is not filed again (the same file received again, or a filing cut short after its
    // ai: row: its take let go); else marked as downloaded, then into Downloads under a free name and listed (place), a
    // ai: tar unpacked there whole (unpack; one that does not is filed as the file), then the take let go
    // ai: (Session.placed). Where Downloads does not take it (it cannot be written, it is full, the system keeps the
    // ai: app out of it), the file stays in the take's folder, listed from there (it opens, marked, and nothing of it
    // ai: is lost), and retry tries again.
    private fun file(k: Kept): Filed {
        if (k.root.isNotEmpty() && list().any { it.root == k.root && it.path != k.file.path }) {
            placed(k.id)
            return Filed(k.root, k.name, "")
        }
        mark(k.file)
        val now = System.currentTimeMillis()
        return try {
            (if (k.type == TAR_TYPE) unpack(k) { folder, files, bytes -> Entry(folder.name, folder.path, k.type, bytes, k.root, now, true, files) } else null)
                ?: place(k.file, k.root) { Entry(it.name, it.path, k.type, k.file.length(), k.root, now, false, 0) }
            placed(k.id)
            Filed(k.root, k.name, "")
        } catch (e: Exception) {
            val why = whyOf(e)
            System.err.println("received: ${k.name} not filed in ${downloads.path}: $why; kept in ${k.file.path}")
            if (k.file.isFile) synchronized(mu) {
                if (read().none { it.path == k.file.path }) add(Entry(k.name, k.file.path, k.type, k.file.length(), k.root, now, false, 0))
            }
            Filed(k.root, k.name, why)
        }
    }

    // ai: Every take in kept/ filed (the caller's thread: the files' one), oldest first: those the session took since,
    // ai: those Downloads did not take (listed from the app's folder) and any a close cut short (the staging it left is
    // ai: made afresh); what each came to. A take not yet whole (its file not in yet) is not listed by the session.
    // ai: Staging in Downloads that no take names any more (one deleted since) goes: only this thread makes staging, so
    // ai: none is under way meanwhile.
    fun retry(): List<Filed> {
        own()
        val takes = kept()
        val roots = takes.map { it.id.substringBefore('-') }.toSet()
        downloads.listFiles()?.filter { f -> STAGING.any { f.name.startsWith(it) } && f.name.substringAfterLast('-') !in roots }
            ?.forEach { runCatching { it.deleteRecursively() } }
        return takes.mapNotNull { k -> runCatching { file(k) }.getOrElse { System.err.println("received: ${k.id} not filed: ${it.message}"); null } }
    }
    // ai: the staging names in Downloads (place, unpack), each followed by its take's 16 hex
    private val STAGING = listOf(".lizard-copying-", ".lizard-unpacking-")

    // ai: One process files at a time: kept/.lock, waited for at a filing's start and then held for the process's life
    // ai: (the system lets it go at the exit or a kill). A launch while a closing one still files (ReceiveState.close
    // ai: waits for its files' thread with the window gone) waits for it rather than redoing its takes beside it, in the
    // ai: same staging. A folder that takes no lock (a filesystem with none) is filed without one.
    private var owned: FileLock? = null
    private fun own() {
        if (owned != null) return
        owned = runCatching {
            FileChannel.open(File(keptDir.apply { mkdirs() }, ".lock").toPath(), StandardOpenOption.CREATE, StandardOpenOption.WRITE).lock()
        }.getOrElse { System.err.println("received: ${keptDir.path}/.lock: ${it.message}; filing without it"); null }
    }

    // ai: Into Downloads whole or not at all, listed (entryOf its new place; moveListed): a rename where the two are one
    // ai: filesystem, else a copy into a hidden file there (the one a copy cut short left, replaced), made durable,
    // ai: marked, then renamed to its free name (a name never holds part of a file) and the folder made durable; the
    // ai: source is the take's, let go after. The entry; a throw where Downloads does not take it (the source stays).
    private fun place(src: File, root: String, entryOf: (File) -> Entry): Entry {
        val dir = writable()
        val dest = free(dir, src.name)
        if (moveListed(src, dest, entryOf(dest))) return entryOf(dest)
        // ai: no copy begun that cannot end (0: the room not known)
        val room = dir.usableSpace
        if (room in 1 until src.length()) throw IOException("${dir.path} has ${Readout.bytes(room)} free, the file is ${Readout.bytes(src.length())}")
        val part = File(dir, STAGING[0] + root.take(16))
        try {
            part.delete()   // ai: a copy cut short (Windows refuses to copy over a hidden file)
            Files.copy(src.toPath(), part.toPath())
            durable(part)
            hide(part, true)
            mark(part)
            if (!moveListed(part, dest, entryOf(dest))) throw IOException("${part.path} could not be renamed to ${dest.name}")
            hide(dest, false)
            durable(dir)
        } catch (e: Exception) { part.delete(); throw e }
        return entryOf(dest)
    }

    // ai: a file's bytes, or a folder's entries (its renames), on the disk before the only other copy goes; a folder's
    // ai: where the system syncs folders (Windows refuses: its renames are the file system's own)
    private fun durable(f: File) {
        if (f.isDirectory) { runCatching { FileChannel.open(f.toPath(), StandardOpenOption.READ).use { it.force(true) } }; return }
        FileChannel.open(f.toPath(), StandardOpenOption.WRITE).use { it.force(true) }
    }

    // ai: Downloads, made where missing; a throw where its permissions refuse a write (File.canWrite: access(W_OK); on
    // ai: Windows a folder's own check is that it exists, so a write's failure says the rest)
    private fun writable(): File {
        val dir = downloads
        if (!dir.isDirectory && !dir.mkdirs()) throw IOException("${dir.path} cannot be made")
        if (!dir.canWrite()) throw AccessDeniedException(dir.path)
        return dir
    }

    // ai: A take's tar unpacked into Downloads whole or not at all: into a hidden folder of its own there (Tar.kt untar,
    // ai: which refuses links, paths out of it and damaged headers; the one an unpack cut short left, removed first), each
    // ai: file marked, then that folder, or the one folder alone at its top, renamed to a free name in the same folder
    // ai: (one rename: the bundle is there whole, or nothing is), its files made durable first and the bundle listed with
    // ai: the rename (entryOf its place, its files and bytes; moveListed). Null where it does not unpack whole (nothing of
    // ai: it left; the tar is then filed as the file); a throw where Downloads cannot hold the folder.
    private fun unpack(k: Kept, entryOf: (File, Int, Long) -> Entry): Entry? {
        val dir = writable()
        val part = File(dir, STAGING[1] + k.root.take(16))
        part.deleteRecursively()
        if (!part.mkdir()) throw IOException("${part.path} cannot be made")
        hide(part, true)
        val made = try { k.file.inputStream().buffered(1 shl 20).use { untar(it, part) } }
            catch (e: Exception) { System.err.println("received: ${k.name} does not unpack: ${e.message}"); part.deleteRecursively(); return null }
        val bytes = made.sumOf { it.length() }   // ai: before the renames, which leave these paths behind
        try { made.forEach { durable(it) } } catch (e: Exception) { part.deleteRecursively(); throw e }
        mark(part)
        val one = part.listFiles().orEmpty().singleOrNull()?.takeIf { it.isDirectory }
        val place = free(dir, one?.name ?: k.name.removeSuffix(".tar").ifEmpty { "files" })
        val e = entryOf(place, made.size, bytes)
        val ok = moveListed(one ?: part, place, e).also { if (it && one == null) hide(place, false) }
        part.deleteRecursively()
        if (!ok) { System.err.println("received: ${k.name} unpacked, but not renamed to ${place.path}"); return null }
        durable(dir)
        return e
    }

    // ai: Windows: a staging file or folder hidden while it is made (a dot alone hides it elsewhere), shown once in place
    private fun hide(f: File, on: Boolean) {
        if (Native.windows) runCatching { Files.setAttribute(f.toPath(), "dos:hidden", on) }
    }

    // ai: why a filing failed, as the line says it
    fun whyOf(e: Throwable): String = when (e) {
        is AccessDeniedException -> "${e.file}: access denied"
        is NoSuchFileException -> "${e.file}: no such file or folder"
        is FileSystemException -> listOfNotNull(e.file, e.reason).joinToString(": ").ifEmpty { e.javaClass.simpleName }
        else -> e.message ?: e.javaClass.simpleName
    }

    // ai: Downloaded-file marks (2026-10-25). A received file's name and bytes are the sender's, whoever shows the code (the
    // ai: root proves the file whole, not who sent it), so each file the app files is marked as a browser marks a
    // ai: download, and the system's checks for files from the internet run when one is opened (open: the desktop's own
    // ai: app, one click from the list).
    // ai: Windows: the Zone.Identifier stream of each file, ZoneId=3 (the Internet zone), which SmartScreen, the
    // ai: Attachment Manager's prompt and Office's Protected View read; written through java.io, which opens a
    // ai: "path:stream" name as NTFS takes it (JDK 17: jdk.io.File.enableADS, true unless set false). A volume with no
    // ai: streams (FAT32, exFAT) takes none, as a browser's download there.
    // ai: macOS: com.apple.quarantine on each file and each folder (an app is a folder), "0081;<seconds, hex>;LIZARD;", the
    // ai: flags a browser outside the App Sandbox writes on a download: 0x0001, downloaded, set and 0x0040, approved by
    // ai: the user, clear, so Gatekeeper assesses the file at its first open and asks before an app runs; set by
    // ai: /usr/bin/xattr, since the JDK's user-defined attributes put every name under "user." (sun.nio.fs
    // ai: UnixUserDefinedFileAttributeView), where Gatekeeper looks for none.
    // ai: Linux has no such mark: received files stay non-executable (0644 under the usual umask: the store and untar make
    // ai: each as a new file, 0666 less the umask, and no mode the sender's tar names is applied), so a desktop opens a
    // ai: received program as a document and runs none.
    // ai: A mark that fails is said in the log; the file is filed whatever the mark does.
    fun mark(f: File) {
        try {
            when {
                Native.windows -> f.walkTopDown().filter { it.isFile }.forEach { zone(it) }
                Native.mac -> quarantine(f.walkTopDown().map { it.absolutePath }.toList())
            }
        } catch (e: Exception) {
            System.err.println("received: ${f.name} not marked as downloaded: ${e.message ?: e.javaClass.simpleName}")
        }
    }
    // ai: Windows' mark of one file: its stream's text
    val ZONE: ByteArray = "[ZoneTransfer]\r\nZoneId=3\r\n".toByteArray(Charsets.US_ASCII)
    fun zone(f: File) = FileOutputStream(File(f.path + ":Zone.Identifier")).use { it.write(ZONE) }
    // ai: macOS' mark: xattr's command for these paths (absolute, so none reads as an option), seconds since 1970
    fun quarantineArgs(paths: List<String>, nowSec: Long): List<String> =
        listOf("/usr/bin/xattr", "-w", "com.apple.quarantine", String.format(Locale.ROOT, "0081;%08x;LIZARD;", nowSec)) + paths
    private fun quarantine(paths: List<String>) {
        val now = System.currentTimeMillis() / 1000
        for (some in paths.chunked(256)) {
            val p = ProcessBuilder(quarantineArgs(some, now)).redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.DISCARD).start()
            val rc = p.waitFor()
            if (rc != 0) throw IOException("xattr ended $rc")
        }
    }

    // ai: An entry gone: its file or folder to the trash where the desktop has one, else deleted; one listed from a take's
    // ai: folder (Downloads did not take it) takes that folder with it, so no retry files it again. Off the UI thread.
    fun delete(e: Entry) {
        synchronized(mu) { write(read().filter { it.path != e.path }) }
        val f = e.file
        val trashed = runCatching { Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.MOVE_TO_TRASH) && Desktop.getDesktop().moveToTrash(f) }
            .getOrDefault(false)
        if (!trashed) f.deleteRecursively()
        val take = f.absoluteFile.parentFile?.parentFile
        if (take?.parentFile?.absolutePath == keptDir.absolutePath) placed(take.name)
    }

    // ai: The desktop's own app for a file (java.awt.Desktop, else xdg-open, open or explorer), on a thread of its own:
    // ai: Desktop's calls can block on the desktop's services
    fun open(f: File) = thread(name = "lizard-open", isDaemon = true) {
        val ok = runCatching { Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.OPEN) && run { Desktop.getDesktop().open(f); true } }
            .getOrDefault(false)
        if (!ok) run(if (Native.windows) listOf("explorer", f.path) else if (Native.mac) listOf("open", f.path) else listOf("xdg-open", f.path))
    }

    // ai: A web page in the desktop's browser (Home's example on video)
    fun browse(url: String) = thread(name = "lizard-open", isDaemon = true) {
        val ok = runCatching { Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE) && run { Desktop.getDesktop().browse(java.net.URI(url)); true } }
            .getOrDefault(false)
        if (!ok) run(if (Native.windows) listOf("explorer", url) else if (Native.mac) listOf("open", url) else listOf("xdg-open", url))
    }

    // ai: The file shown in its folder, selected: Explorer's /select, Finder's open -R, else the file manager's
    // ai: org.freedesktop.FileManager1 ShowItems over D-Bus (gdbus), else the folder opened (xdg-open)
    fun show(f: File) = thread(name = "lizard-open", isDaemon = true) {
        when {
            Native.windows -> run(listOf("explorer", "/select,", f.path))
            Native.mac -> run(listOf("open", "-R", f.path))
            else -> {
                // ai: file:///path, the form the FileManager1 spec shows (File.toURI gives file:/path); a quote escaped
                // ai: inside GVariant's quotes
                val uri = f.absoluteFile.toPath().toUri().toASCIIString().replace("'", "%27")
                val asked = run(listOf("gdbus", "call", "--session", "--dest", "org.freedesktop.FileManager1", "--object-path", "/org/freedesktop/FileManager1",
                    "--method", "org.freedesktop.FileManager1.ShowItems", "['$uri']", ""))
                if (!asked) run(listOf("xdg-open", (f.parentFile ?: f).path))
            }
        }
    }

    // ai: a command run to its end, its output let go; whether it said 0 (Explorer says 1 when it worked: started is enough there)
    private fun run(cmd: List<String>): Boolean = runCatching {
        val p = ProcessBuilder(cmd).redirectOutput(ProcessBuilder.Redirect.DISCARD).redirectError(ProcessBuilder.Redirect.DISCARD).start()
        if (Native.windows) true else p.waitFor() == 0
    }.getOrElse { System.err.println("received: ${cmd.first()}: ${it.message}"); false }
}
