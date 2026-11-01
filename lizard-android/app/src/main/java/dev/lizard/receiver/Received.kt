package dev.lizard.receiver

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.util.Log
import android.webkit.MimeTypeMap
import androidx.core.content.FileProvider
import dev.lizard.Session
import dev.lizard.tar.TAR_TYPE
import dev.lizard.tar.untar
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import dev.lizard.ui.*

// ai: The received files (2026-10-31; the app's own library in filesDir/library before, the desktop's Received.kt the
// ai: model): a verified file is taken out of the receiver's store by the receive session into its data folder's kept/
// ai: (liblizard/core/rx/session.h: kept/<id>/ with filing.json and file/<name>; Session.kept lists them), then placed
// ai: here into the phone's Downloads through MediaStore (RELATIVE_PATH Download/, no permission from Android 10) and
// ai: its take let go (Session.placed). A tar (several files or a folder sent as one, Tar.kt) is unpacked into a
// ai: folder there whole (one folder alone at its top is the bundle's, else the tar's name less .tar; a free name, the
// ai: folder's files hidden as pending until the last is in), one that does not unpack whole placed as the file. What
// ai: Downloads does not take stays in the data folder, listed and opened from there (the FileProvider's "session"
// ai: path), and is placed again at the next pass (retry): a file is never only in a store that lets it go. A take is
// ai: placed when its file is in Downloads, made durable, and listed: its row written before its take is let go (a
// ai: copy whose row cannot be written is taken back out of Downloads), so a placing cut short leaves the take, which
// ai: the next pass lets go once its root is listed. The list is <data>/received.json (entries name, type, size, root,
// ai: at, uri for a file, folder and items {path, uri, size} for a bundle, local for one Downloads did not take), newest
// ai: first, written durably whole; an entry whose file MediaStore says is gone (the user deleted it there) is dropped
// ai: from it, but not one MediaStore did not answer for, and a list that does not read is never written over (one that
// ai: does not parse is set aside whole as received.json.bad); the same root placed again replaces its entry. Every
// ai: read and write of the list is under one lock; list and retry query MediaStore, so they run off the main thread.
// ai: The files received before 2026-10-31 (filesDir/library, Library.kt's <id>/<name> and .meta.json) are moved into
// ai: <data>/library at the first start (MainActivity.carryOver) and placed by the same pass, each removed from there
// ai: once it is in Downloads.
class Received(private val ctx: Context, private val data: File) {
    // ai: a bundle's file: its path in the folder, its MediaStore URI (or a local file's), its size
    data class Item(val path: String, val uri: String, val size: Long)
    data class Entry(val name: String, val type: String, val size: Long, val root: String, val at: Long, val uri: String, val folder: String,
                     val items: List<Item>, val local: String) {
        val bundle get() = folder.isNotEmpty() || items.isNotEmpty()
        val files get() = items.size
        // ai: the key a row is kept by: the root, else the file's place
        val id get() = root.ifEmpty { uri.ifEmpty { local } }
        // ai: the header's media type, else the name's extension's, else bytes
        val mime: String get() = type.takeIf { it.isNotEmpty() && it != TAR_TYPE }.orEmpty().ifEmpty { mimeOf(name) }
    }

    private val mu = Any()
    private val listFile get() = File(data, "received.json")
    private val keptDir get() = File(data, "kept")
    private val oldLibrary get() = File(data, "library")
    private val resolver get() = ctx.contentResolver

    // ai: the URI another app opens a file by: MediaStore's for a file in Downloads, the FileProvider's for one kept in
    // ai: the data folder
    fun uriOf(u: String): Uri = if (u.startsWith("content:")) Uri.parse(u) else FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", File(u))

    // ai: the entries whose files are there, newest first; a list that does not read gives none and is left as it is
    fun list(): List<Entry> = synchronized(mu) {
        val all = runCatching { read() }.getOrElse { Log.w(TAG, "received: ${listFile.path}: ${it.message}"); return emptyList() }
        val here = all.filter { present(it) }
        if (here.size != all.size) runCatching { write(here) }.onFailure { Log.w(TAG, "received: ${listFile.path}: ${it.message}") }
        here.sortedByDescending { it.at }
    }

    // ai: whether an entry's file may still be there (a MediaStore row of this app's, or the local file): gone only where
    // ai: MediaStore answers with no row; a query that throws or gives no answer (the media provider restarting) keeps it
    private fun present(e: Entry): Boolean = when {
        e.local.isNotEmpty() -> File(e.local).exists()
        e.uri.isNotEmpty() -> exists(e.uri)
        else -> e.items.any { exists(it.uri) }
    }
    private fun exists(u: String): Boolean = if (!u.startsWith("content:")) File(u).exists() else runCatching {
        resolver.query(Uri.parse(u), arrayOf(MediaStore.MediaColumns._ID), null, null, null)?.use { it.moveToFirst() } ?: true
    }.getOrDefault(true)

    // ai: the list as written: none where there is no list yet; a read that fails throws (nothing is written over the list
    // ai: then); a list that does not parse (a zero-length one a power loss left, before its writes were made durable) is
    // ai: set aside whole beside it as received.json.bad, said in the log, and a new list begun
    private fun read(): List<Entry> {
        val f = listFile
        if (!f.exists()) return emptyList()
        val text = f.readText()
        val a = try { JSONArray(text) } catch (x: JSONException) {
            val bad = File(data, "received.json.bad")
            bad.delete()
            if (!f.renameTo(bad)) throw IOException("${f.path} does not parse and could not be set aside")
            Log.w(TAG, "received: ${f.path} did not parse (${text.length} B): set aside as ${bad.path}")
            return emptyList()
        }
        return (0 until a.length()).mapNotNull { i ->
            val o = a.optJSONObject(i) ?: return@mapNotNull null
            val its = o.optJSONArray("items")
            val items = if (its == null) emptyList() else (0 until its.length()).mapNotNull { k ->
                its.optJSONObject(k)?.let { Item(it.optString("path"), it.optString("uri"), it.optLong("size")) } }
            Entry(o.optString("name"), o.optString("type"), o.optLong("size"), o.optString("root"), o.optLong("at"), o.optString("uri"),
                o.optString("folder"), items, o.optString("local"))
        }
    }

    // ai: written whole beside it, made durable, and renamed over it, so a crash or a power loss leaves the list as it was
    private fun write(es: List<Entry>) {
        val a = JSONArray()
        for (e in es) a.put(JSONObject().put("name", e.name).put("type", e.type).put("size", e.size).put("root", e.root).put("at", e.at)
            .put("uri", e.uri).put("folder", e.folder).put("local", e.local)
            .put("items", JSONArray().apply { for (it in e.items) put(JSONObject().put("path", it.path).put("uri", it.uri).put("size", it.size)) }))
        data.mkdirs()
        val tmp = File(data, "received.json.tmp")
        FileOutputStream(tmp).use { o -> o.write(a.toString().toByteArray()); o.fd.sync() }
        if (!tmp.renameTo(listFile)) throw IOException("${listFile.path} could not be written")
    }

    private fun add(e: Entry) = synchronized(mu) { write(read().filter { (it.root != e.root || e.root.isEmpty()) && !(e.local.isNotEmpty() && it.local == e.local) } + e) }
    // ai: an entry placed into Downloads listed; where its row cannot be written, its files taken back out of Downloads
    // ai: (placed means listed: the take stays, placed again by a later pass)
    private fun listed(e: Entry) {
        try { add(e) } catch (x: Exception) {
            if (e.uri.startsWith("content:")) runCatching { resolver.delete(Uri.parse(e.uri), null, null) }
            for (it in e.items) if (it.uri.startsWith("content:")) runCatching { resolver.delete(Uri.parse(it.uri), null, null) }
            throw x
        }
    }

    // ai: What a placing came to: the root and name, and "" where the file is in Downloads, else why it is not (it is in
    // ai: the data folder then, listed from there)
    class Filed(val root: String, val name: String, val why: String)

    // ai: a take as the session lists it (Session.kept)
    private class Take(val id: String, val name: String, val type: String, val root: String, val file: File)
    private fun takes(): List<Take> = runCatching {
        val a = JSONArray(Session.kept(data.path))
        (0 until a.length()).mapNotNull { i ->
            val o = a.optJSONObject(i) ?: return@mapNotNull null
            Take(o.optString("id"), o.optString("name"), o.optString("type"), o.optString("root"), File(o.optString("path"))).takeIf { it.id.isNotEmpty() && it.name.isNotEmpty() }
        }
    }.getOrDefault(emptyList())
    private fun placed(id: String) {
        val why = Session.placed(data.path, id)
        if (why.isNotEmpty()) Log.w(TAG, "received: kept/$id: $why")
    }

    // ai: Every take in kept/ placed, oldest first, then the files received before 2026-10-31 (<data>/library): what each
    // ai: came to. One pass at a time (the caller's one thread, MainActivity's filer).
    fun retry(): List<Filed> {
        val out = ArrayList<Filed>()
        for (k in takes()) out += runCatching { file(k) }.getOrElse { Log.w(TAG, "received: ${k.id} not placed: ${it.message}"); Filed(k.root, k.name, it.message.orEmpty()) }
        for (d in oldLibrary.listFiles().orEmpty().filter { it.isDirectory }) out += runCatching { carried(d) }.getOrElse { Filed("", d.name, it.message.orEmpty()) }
        return out
    }

    // ai: A take placed: a root a list entry holds in Downloads is not placed again (the same file received again, or a
    // ai: placing cut short after its row: its take let go); else into Downloads (a tar unpacked there whole), the entry
    // ai: listed, then the take let go (its file with it). Where Downloads does not take it the file stays in the take's
    // ai: folder, listed from there.
    private fun file(k: Take): Filed {
        if (k.root.isNotEmpty() && list().any { it.root == k.root && it.local != k.file.path }) {
            placed(k.id)
            return Filed(k.root, k.name, "")
        }
        val now = System.currentTimeMillis()
        return try {
            val e = (if (k.type == TAR_TYPE) unpack(k.file, k.name, k.root, now) else null)
                ?: placeFile(k.file, k.name, k.type).let { (uri, size) -> Entry(k.name, k.type, size, k.root, now, uri, "", emptyList(), "") }
            listed(e)
            placed(k.id)
            Filed(k.root, k.name, "")
        } catch (e: Exception) {
            val why = e.message ?: e.javaClass.simpleName
            Log.w(TAG, "received: ${k.name} not placed in Downloads: $why; kept in ${k.file.path}")
            if (k.file.isFile) synchronized(mu) {
                if (read().none { it.local == k.file.path }) add(Entry(k.name, k.type, k.file.length(), k.root, now, "", "", emptyList(), k.file.path))
            }
            Filed(k.root, k.name, why)
        }
    }

    // ai: A file received before 2026-10-31 (Library.kt's folder: the file or the bundle's folder beside .meta.json)
    // ai: placed into Downloads as a take is, its folder removed once it is there; one Downloads does not take listed from
    // ai: where it is
    private fun carried(d: File): Filed {
        val m = runCatching { JSONObject(File(d, ".meta.json").readText()) }.getOrNull()
        val stored = m?.optString("stored").orEmpty()
        val f = if (stored.isNotEmpty()) File(d, stored) else d.listFiles()?.firstOrNull { !it.name.startsWith(".") }
        if (f == null || !f.exists()) { d.deleteRecursively(); return Filed("", d.name, "") }
        val name = m?.optString("name").orEmpty().ifEmpty { f.name }
        val root = m?.optString("root").orEmpty()
        val type = m?.optString("type").orEmpty()
        val at = m?.optLong("at")?.takeIf { it > 0 } ?: f.lastModified()
        if (root.isNotEmpty() && list().any { it.root == root && it.local.isEmpty() }) { d.deleteRecursively(); return Filed(root, name, "") }
        return try {
            val e = if (f.isDirectory) placeFolder(f, f.name, root, at, type)
                else placeFile(f, name, type).let { (uri, size) -> Entry(name, type, size, root, at, uri, "", emptyList(), "") }
            listed(e)
            d.deleteRecursively()
            Filed(root, name, "")
        } catch (x: Exception) {
            val why = x.message ?: x.javaClass.simpleName
            Log.w(TAG, "received: ${f.name} (received before) not placed in Downloads: $why")
            synchronized(mu) {
                if (read().none { it.local == f.path }) {
                    val items = if (f.isDirectory) f.walkTopDown().filter { it.isFile }.map { Item(it.relativeTo(f).invariantSeparatorsPath, it.path, it.length()) }.toList() else emptyList()
                    add(Entry(name, type, if (f.isDirectory) items.sumOf { it.size } else f.length(), root, at, "", if (f.isDirectory) f.name else "", items, f.path))
                }
            }
            Filed(root, name, why)
        }
    }

    // ai: One file into Downloads through MediaStore: inserted pending (hidden from other apps), its bytes copied and made
    // ai: durable, then shown; MediaStore gives a name taken a free one ("name (1).ext"). Its URI and size; a throw where
    // ai: Downloads does not take it (nothing of it left there). The source is the caller's to let go once it is listed.
    private fun placeFile(src: File, name: String, type: String, sub: String = "", pending: MutableList<Uri>? = null): Pair<String, Long> {
        val v = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, safeName(name))
            put(MediaStore.MediaColumns.MIME_TYPE, type.ifEmpty { mimeOf(name) })
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/" + sub)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v) ?: throw IOException("Downloads refused $name")
        try {
            val fd = resolver.openFileDescriptor(uri, "w") ?: throw IOException("Downloads refused $name")
            ParcelFileDescriptor.AutoCloseOutputStream(fd).use { o -> src.inputStream().use { it.copyTo(o, 1 shl 16) }; o.fd.sync() }
            if (pending == null) resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
            else pending += uri
        } catch (e: Throwable) {
            runCatching { resolver.delete(uri, null, null) }
            throw e
        }
        return uri.toString() to src.length()
    }

    // ai: A folder's files into Downloads/<a free name>/..., each pending until the last is in, then all shown (the
    // ai: bundle there whole, or none of it: what was inserted is deleted at a failure). The folder's name is free where
    // ai: no file of this app's is under it and (Android 11 and up, where the folder is seen) no folder of that name is
    // ai: there. The entry; the source folder is left to the caller.
    private fun placeFolder(dir: File, name: String, root: String, at: Long, type: String): Entry {
        val folder = freeFolder(safeName(name))
        val files = dir.walkTopDown().filter { it.isFile }.sortedBy { it.relativeTo(dir).invariantSeparatorsPath }.toList()
        val pending = ArrayList<Uri>()
        val items = ArrayList<Item>()
        try {
            for (f in files) {
                val rel = f.relativeTo(dir).invariantSeparatorsPath
                val sub = (folder + "/" + rel.substringBeforeLast('/', "")).trimEnd('/') + "/"
                val (uri, size) = placeFile(f, f.name, "", sub, pending)
                items += Item(rel, uri, size)
            }
            for (u in pending) resolver.update(u, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
        } catch (e: Throwable) {
            for (u in pending) runCatching { resolver.delete(u, null, null) }
            throw e
        }
        return Entry(folder, type.ifEmpty { TAR_TYPE }, items.sumOf { it.size }, root, at, "", Environment.DIRECTORY_DOWNLOADS + "/" + folder, items, "")
    }

    private fun freeFolder(name: String): String {
        @Suppress("DEPRECATION") val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        fun taken(n: String): Boolean {
            if (Build.VERSION.SDK_INT >= 30 && File(downloads, n).exists()) return true
            return runCatching {
                resolver.query(MediaStore.Downloads.EXTERNAL_CONTENT_URI, arrayOf(MediaStore.MediaColumns._ID),
                    "${MediaStore.MediaColumns.RELATIVE_PATH} LIKE ?", arrayOf(Environment.DIRECTORY_DOWNLOADS + "/" + n + "/%"), null)?.use { it.count > 0 } ?: false
            }.getOrDefault(false)
        }
        if (!taken(name)) return name
        var k = 1
        while (taken("$name ($k)")) k++
        return "$name ($k)"
    }

    // ai: A take's tar unpacked (Tar.kt untar, which refuses links, paths out of it and damaged headers) into a folder of
    // ai: its own beside it, then placed as a folder: one folder alone at its top is the bundle's, else the tar's name
    // ai: less .tar. Null where it does not unpack whole (nothing of it left; the tar is then placed as the file). The
    // ai: tar stays, its take's to let go.
    private fun unpack(tar: File, name: String, root: String, at: Long): Entry? {
        val part = File(tar.parentFile?.parentFile ?: data, ".unpacking").apply { deleteRecursively() }
        try {
            try { tar.inputStream().buffered(1 shl 20).use { untar(it, part) } }
            catch (e: Exception) { Log.w(TAG, "unpack: ${tar.name}: ${e.message}"); return null }
            val top = part.listFiles().orEmpty()
            val one = top.singleOrNull()?.takeIf { it.isDirectory }
            return placeFolder(one ?: part, one?.name ?: name.removeSuffix(".tar").ifEmpty { "files" }, root, at, TAR_TYPE)
        } finally { part.deleteRecursively() }
    }

    // ai: An entry gone: its files deleted from Downloads (this app's own rows), or its local file or folder; one listed
    // ai: from a take's folder takes that folder with it. Its row goes only once its files have: a delete MediaStore
    // ai: refused keeps it (a throw, said by the caller). Off the main thread.
    fun delete(e: Entry) {
        val uris = (listOf(e.uri) + e.items.map { it.uri }).filter { it.startsWith("content:") }
        var left = uris.count { u -> !(runCatching { resolver.delete(Uri.parse(u), null, null) > 0 }.getOrDefault(false) || !exists(u)) }
        if (e.local.isNotEmpty()) {
            val f = File(e.local)
            if (!f.deleteRecursively()) left++
            else {
                val take = f.absoluteFile.parentFile?.parentFile
                if (take?.parentFile?.absolutePath == keptDir.absolutePath) placed(take.name)
                val lib = f.absoluteFile.parentFile
                if (lib?.parentFile?.absolutePath == oldLibrary.absolutePath) lib.deleteRecursively()
            }
        }
        if (left > 0) throw IOException("${e.name}: $left of its files not deleted")
        synchronized(mu) { write(read().filter { it.id != e.id || it.local != e.local }) }
    }

    companion object {
        // ai: a name's media type by its extension, else bytes
        fun mimeOf(name: String): String =
            MimeTypeMap.getSingleton().getMimeTypeFromExtension(name.substringAfterLast('.', "").lowercase()) ?: "application/octet-stream"
        // ai: a name MediaStore takes: no separator, no control character, not "." or ".."
        fun safeName(name: String): String {
            val n = name.map { if (it == '/' || it == '\\' || it.code < 0x20 || it.code == 0x7f) '_' else it }.joinToString("")
            return if (n.isEmpty() || n == "." || n == "..") "file" else n
        }
    }
}
