package dev.lizard.receiver

import dev.lizard.tar.TarWriter
import java.io.File
import java.io.FileOutputStream
import java.io.FilterOutputStream
import java.io.OutputStream
import java.util.zip.Deflater
import java.util.zip.GZIPOutputStream

// ai: A replay as one file (2026-10-03): the run's folder as a ustar tar (liblizard/kotlin/Tar.kt TarWriter, since
// ai: 2026-10-16; the same headers before: meta.json, stats.jsonl, then NNNN.gray, the names flat), under gzip at
// ai: level 1. Light is all lossless buys here: a replay's frames are mostly the code, noise-like by design, and on the
// ai: S26's 2:1 frames gzip -1 leaves 92.1% of their bytes, gzip -6 92.5%, zstd -19 90.9%, xz -1 87.3%. Pure JVM (no
// ai: Android): into a stream (MediaStore's, Android 10), or a file written through a .part renamed at the end (Android
// ai: 11 and up); the archive's bytes.
internal fun packReplay(dir: File, out: OutputStream): Long {
    val files = dir.listFiles()?.filter { it.isFile }.orEmpty()
        .sortedWith(compareBy({ it.name != "meta.json" }, { it.name != "stats.jsonl" }, { it.name }))
    require(files.any { it.name == "meta.json" }) { "the replay has no meta.json" }
    val mtime = System.currentTimeMillis() / 1000
    val counted = Counting(out)
    object : GZIPOutputStream(counted, 1 shl 16) { init { def.setLevel(Deflater.BEST_SPEED) } }.use { gz ->
        val w = TarWriter(gz)
        for (f in files) f.inputStream().use { w.file(f.name, f.length(), mtime, it) }
        w.finish()
    }
    return counted.n
}

internal fun packReplay(dir: File, dest: File): Long {
    val part = File(dest.parentFile, dest.name + ".part")
    try {
        val n = FileOutputStream(part).use { packReplay(dir, it) }
        if (dest.exists() && !dest.delete()) error("${dest.name} could not be replaced")
        if (!part.renameTo(dest)) error("${part.name} could not be renamed")
        return n
    } catch (e: Throwable) {
        part.delete()
        throw e
    }
}

private class Counting(out: OutputStream) : FilterOutputStream(out) {
    var n = 0L
    override fun write(b: Int) { out.write(b); n++ }
    override fun write(b: ByteArray, off: Int, len: Int) { out.write(b, off, len); n += len }
}
