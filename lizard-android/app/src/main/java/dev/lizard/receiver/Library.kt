package dev.lizard.receiver

import android.webkit.MimeTypeMap
import org.json.JSONObject
import java.io.File

// ai: The received files (2026-10-01), kept in the app's own list: filesDir/library/<id>/ holding the file under its
// ai: own name and .meta.json ({ name, type, size, root, at }),
// ai: the id the first 16 hex digits of its BLAKE3 root, so a file received again replaces its entry. A verified file is
// ai: moved here from the native store (Engine.keep: the same filesystem, so a rename) and every later Open, Share,
// ai: Save and Delete reads it here; the store's own copy is gone, and a receiver's rebuild leaves this alone. Open and Share hand it out through a FileProvider (AndroidManifest.xml,
// ai: res/xml/file_paths.xml). The web keeps the same list in its origin's OPFS (lizard-web/library.mjs).
class Library(private val dir: File) {
    data class Entry(val id: String, val name: String, val type: String, val size: Long, val root: String, val at: Long, val file: File) {
        // ai: the header's media type, else the name's extension's, else bytes
        val mime: String get() = type.ifEmpty {
            MimeTypeMap.getSingleton().getMimeTypeFromExtension(name.substringAfterLast('.', "").lowercase()) ?: "application/octet-stream"
        }
    }

    fun list(): List<Entry> = (dir.listFiles() ?: emptyArray()).filter { it.isDirectory }.mapNotNull { d ->
        try {
            val m = JSONObject(File(d, META).readText())
            val f = File(d, m.getString("stored"))
            if (!f.isFile) null else Entry(d.name, m.optString("name"), m.optString("type"), m.optLong("size", f.length()), m.optString("root"), m.optLong("at"), f)
        } catch (_: Exception) { null }
    }.sortedByDescending { it.at }

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
        // ai: a name the filesystem takes, as the native store makes it (liblizard/core/rx/xfer_rx.cpp safeName: no separator, no
        // ai: control character, no "." or ".."), and not the meta file's
        fun safeName(name: String): String {
            val n = name.map { if (it == '/' || it == '\\' || it.code < 0x20 || it.code == 0x7f) '_' else it }.joinToString("")
            return when { n.isEmpty() || n == "." || n == ".." -> "file"; n == META -> "_$n"; else -> n }
        }
    }
}
