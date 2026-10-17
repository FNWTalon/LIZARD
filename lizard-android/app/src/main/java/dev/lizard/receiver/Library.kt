package dev.lizard.receiver

import android.util.Log
import android.webkit.MimeTypeMap
import dev.lizard.tar.untar
import org.json.JSONObject
import java.io.File

// ai: The received files (2026-10-01), kept in the app's own list: filesDir/library/<id>/ holding the file under its
// ai: own name and .meta.json ({ name, type, size, root, at }),
// ai: the id the first 16 hex digits of its BLAKE3 root, so a file received again replaces its entry. A verified file is
// ai: moved here from the native store (Engine.keep: the same filesystem, so a rename) and every later Open, Share,
// ai: Save and Delete reads it here; the store's own copy is gone, and a receiver's rebuild leaves this alone. Open and Share hand it out through a FileProvider (AndroidManifest.xml,
// ai: res/xml/file_paths.xml). The web keeps the same list in its origin's OPFS (lizard-web/library.mjs).
// ai: A bundle (2026-10-16: several files or a folder sent as one tar, liblizard/kotlin/Tar.kt) is unpacked where it
// ai: was kept (unpack), its entry the folder: `bundle`, its files' count and their bytes in .meta.json.
class Library(private val dir: File) {
    data class Entry(val id: String, val name: String, val type: String, val size: Long, val root: String, val at: Long, val file: File,
                     val bundle: Boolean = false, val files: Int = 0) {
        // ai: the header's media type, else the name's extension's, else bytes
        val mime: String get() = type.ifEmpty { mimeOf(name) }
        // ai: a bundle's files under its folder, each with its path there, in the order of their paths
        fun contents(): List<Pair<String, File>> =
            file.walkTopDown().filter { it.isFile }.map { it.relativeTo(file).invariantSeparatorsPath to it }.sortedBy { it.first }.toList()
    }

    fun list(): List<Entry> = (dir.listFiles() ?: emptyArray()).filter { it.isDirectory }.mapNotNull { d ->
        try {
            val m = JSONObject(File(d, META).readText())
            val f = File(d, m.getString("stored"))
            val bundle = m.optBoolean("bundle")
            if (if (bundle) !f.isDirectory else !f.isFile) null
            else Entry(d.name, m.optString("name"), m.optString("type"), m.optLong("size", f.length()), m.optString("root"), m.optLong("at"), f, bundle, m.optInt("files"))
        } catch (_: Exception) { null }
    }.sortedByDescending { it.at }

    // ai: A received tar unpacked beside it (off the main thread: a file's whole bytes): into a folder of its own, then
    // ai: that folder renamed into place, the tar deleted and the entry written. One folder alone at the archive's top
    // ai: (a folder sent as itself) is the bundle's folder; anything else goes under the tar's name less ".tar". A tar
    // ai: that does not unpack whole (a link, a path out of its folder, a damaged header: Tar.kt untar) leaves nothing
    // ai: of it, and the caller keeps the tar as a file. The entry, or null.
    fun unpack(tar: File, name: String, root: String): Entry? {
        val d = tar.parentFile ?: return null
        val part = File(d, ".unpacking").apply { deleteRecursively() }
        val made = try { tar.inputStream().buffered(1 shl 20).use { untar(it, part) } }
            catch (e: Exception) { Log.w(Engine.TAG, "unpack: ${tar.name}: ${e.message}"); part.deleteRecursively(); return null }
        val bytes = made.sumOf { it.length() }   // ai: before the renames, which leave these paths behind
        val top = part.listFiles().orEmpty()
        val stem = safeName(name.removeSuffix(".tar").ifEmpty { "files" })
        val folder = if (top.size == 1 && top[0].isDirectory) File(d, safeName(top[0].name)).takeIf { it != tar && it.name != part.name && top[0].renameTo(it) }
            else null
        val home = folder ?: File(d, stem).takeIf { it != tar && part.renameTo(it) }
        if (home == null) { part.deleteRecursively(); return null }
        part.deleteRecursively()
        tar.delete()
        File(d, META).writeText(JSONObject().put("name", home.name).put("type", TAR_MEDIA).put("size", bytes).put("root", root)
            .put("at", System.currentTimeMillis()).put("stored", home.name).put("bundle", true).put("files", made.size).put("sent", name).toString())
        return list().firstOrNull { it.id == d.name }
    }

    // ai: Where a verified file goes: its folder emptied first (the same file received again replaces its entry). The
    // ai: caller moves the file there, then calls kept().
    fun dest(root: String, name: String): File {
        val d = File(dir, root.take(16))
        d.deleteRecursively()
        d.mkdirs()
        return File(d, safeName(name))
    }
    fun kept(file: File, name: String, type: String, size: Long, root: String) {
        File(file.parentFile, META).writeText(JSONObject().put("name", name).put("type", type).put("size", size).put("root", root)
            .put("at", System.currentTimeMillis()).put("stored", file.name).toString())
    }
    fun delete(e: Entry) { File(dir, e.id).deleteRecursively() }
    fun deleteAll() { dir.listFiles()?.forEach { it.deleteRecursively() } }

    companion object {
        private const val META = ".meta.json"
        const val TAR_MEDIA = dev.lizard.tar.TAR_TYPE
        // ai: a name's media type by its extension, else bytes
        fun mimeOf(name: String): String =
            MimeTypeMap.getSingleton().getMimeTypeFromExtension(name.substringAfterLast('.', "").lowercase()) ?: "application/octet-stream"
        // ai: a name the filesystem takes, as the native store makes it (liblizard/core/rx/xfer_rx.cpp safeName: no separator, no
        // ai: control character, no "." or ".."), and not the meta file's
        fun safeName(name: String): String {
            val n = name.map { if (it == '/' || it == '\\' || it.code < 0x20 || it.code == 0x7f) '_' else it }.joinToString("")
            return when { n.isEmpty() || n == "." || n == ".." -> "file"; n == META -> "_$n"; else -> n }
        }
    }
}
