package dev.lizard.receiver

import org.json.JSONObject
import java.io.File

// ai: Transfers begun and not finished (2026-10-22): the receiver's folders under filesDir/received/lizard-xfer
// ai: (liblizard/core/rx/store.h fileStore), one a transfer by its root's first 16 hex digits, each holding its header
// ai: and meta.json ({ name, type, length, chunks, verified, root, fraction, sent, sentIn }). A receiver resumes one when
// ai: its file is in the light again; Home lists them to continue or delete. The web keeps the same in its origin's OPFS
// ai: (lizard-web/library.mjs unfinished).
class Partials(private val dir: File) {
    data class Entry(val id: String, val name: String, val length: Long, val fraction: Double, val root: String, val at: Long, val dir: File)

    // ai: newest first, at the time its meta.json was last written; a folder with no header or no readable meta.json is
    // ai: left out (a receiver opening the folder removes the first kind)
    fun list(): List<Entry> = (dir.listFiles() ?: emptyArray()).filter { it.isDirectory && File(it, "header").isFile }.mapNotNull { d ->
        try {
            val f = File(d, "meta.json")
            val m = JSONObject(f.readText())
            Entry(d.name, m.optString("name"), m.optLong("length"), m.optDouble("fraction", 0.0), m.optString("root"), f.lastModified(), d)
        } catch (_: Exception) { null }
    }.sortedByDescending { it.at }
}
