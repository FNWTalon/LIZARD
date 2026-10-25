package dev.lizard.desktop

import java.io.File

// ai: Transfers begun and not finished (2026-10-25, the PC receiver; the Android app's Partials.kt, the web's
// ai: library.mjs unfinished): the receiver's folders under Native.data/store/lizard-xfer (liblizard/core/rx/store.h
// ai: fileStore), one a transfer by its root's first 16 hex digits, each holding its header and meta.json (name, type,
// ai: length, chunks, verified, root, fraction, sent, sentIn). A receiver on that store resumes one when its file is in
// ai: the light again; Home lists them to continue or delete. A folder with no header, or no readable meta.json, is
// ai: left out (a receiver opening the store removes the first kind); newest meta.json first.
object Partials {
    class Entry(val id: String, val name: String, val length: Long, val fraction: Double, val root: String, val at: Long, val dir: File)

    val dir: File get() = File(Native.data, "store/lizard-xfer")

    fun list(): List<Entry> = (dir.listFiles() ?: emptyArray()).filter { it.isDirectory && File(it, "header").isFile }.mapNotNull { d ->
        val f = File(d, "meta.json")
        val m = runCatching { parse(f.readText()) }.getOrNull() ?: return@mapNotNull null
        Entry(d.name, m.str("name"), m.num("length").toLong(), m.num("fraction"), m.str("root"), f.lastModified(), d)
    }.sortedByDescending { it.at }
}
