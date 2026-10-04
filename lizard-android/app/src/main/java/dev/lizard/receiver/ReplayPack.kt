package dev.lizard.receiver

import java.io.File
import java.io.FileOutputStream
import java.io.FilterOutputStream
import java.io.OutputStream
import java.util.zip.Deflater
import java.util.zip.GZIPOutputStream

// ai: A replay as one file (2026-10-03): the run's folder as a ustar tar in the web's layout and headers
// ai: (lizard-web/tar.mjs; recv.mjs saveRun: meta.json, stats.jsonl, then NNNN.gray, the names flat), under gzip at
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
        val zeros = ByteArray(1024)
        for (f in files) {
            val size = f.length()
            gz.write(tarHeader(f.name, size, mtime))
            f.inputStream().use { it.copyTo(gz, 1 shl 20) }
            val pad = ((512 - size % 512) % 512).toInt()
            if (pad > 0) gz.write(zeros, 0, pad)
        }
        gz.write(zeros)   // ai: the end of the archive, two zero blocks
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

// ai: tar.mjs header: the name (under 100 characters), mode 644, a plain file, size and mtime in octal, "ustar" and
// ai: version "00", the checksum taken with its own field as spaces and written as six octal digits, a NUL and a space
private fun tarHeader(name: String, size: Long, mtime: Long): ByteArray {
    require(name.length <= 99) { "tar name too long: $name" }
    val h = ByteArray(512)
    fun put(off: Int, s: String) = s.toByteArray(Charsets.US_ASCII).copyInto(h, off)
    put(0, name)
    put(100, "0000644\u0000")
    put(108, "0000000\u0000")
    put(116, "0000000\u0000")
    put(124, java.lang.Long.toOctalString(size).padStart(11, '0') + "\u0000")
    put(136, java.lang.Long.toOctalString(mtime).padStart(11, '0') + "\u0000")
    put(148, "        ")
    put(156, "0")
    put(257, "ustar\u0000" + "00")
    val sum = h.sumOf { it.toInt() and 255 }
    put(148, Integer.toOctalString(sum).padStart(6, '0') + "\u0000 ")
    return h
}
