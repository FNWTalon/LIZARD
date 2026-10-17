package dev.lizard.tar

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream

// ai: Several files or a folder as one POSIX tar, the way a sender bundles them into one transfer (the transfer's zstd
// ai: compresses each 4 MiB chunk of the tar as of any file), and the reader a receiver unpacks one with. Plain JVM, no
// ai: Android: the Android app and the desktop sender both compile this file (each build adds liblizard/kotlin to its
// ai: sources). The web's twin is lizard-web/tar.mjs.
// ai:
// ai: The writer: ustar headers (name, and the prefix field where a path is long), a pax extended header (typeflag x)
// ai: carrying `path` where a path fits neither field or is not plain ASCII and `size` past ustar's 8 GiB, folders as
// ai: their own entries (typeflag 5, so an empty folder survives), files mode 644 and folders 755, owner 0, the
// ai: archive ended by two zero blocks. GNU tar, bsdtar and Python's tarfile read it.

const val TAR_TYPE = "application/x-tar"
private const val BLOCK = 512
private val ZEROS = ByteArray(2 * BLOCK)

class TarWriter(private val out: OutputStream) {
    /** The archive's bytes so far. */
    var written = 0L
        private set

    private fun put(b: ByteArray, off: Int = 0, len: Int = b.size) { out.write(b, off, len); written += len }
    private fun pad(n: Long) { val p = ((BLOCK - n % BLOCK) % BLOCK).toInt(); if (p > 0) put(ZEROS, 0, p) }

    /** A folder; `path` relative, '/' between its parts. */
    fun dir(path: String, mtimeSec: Long) = entry(path.trimEnd('/') + "/", 0, mtimeSec, '5', "0000755")

    /** A file of exactly `size` bytes read from `src`; a source that ends short or runs long is an error. */
    fun file(path: String, size: Long, mtimeSec: Long, src: InputStream) {
        entry(path, size, mtimeSec, '0', "0000644")
        val buf = ByteArray(1 shl 20)
        var left = size
        while (left > 0) {
            val n = src.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
            if (n < 0) throw IOException("$path ended $left bytes short of its $size")
            put(buf, 0, n); left -= n
        }
        if (src.read() >= 0) throw IOException("$path is longer than its $size bytes (it changed while bundled)")
        pad(size)
    }

    /** The end of the archive: two zero blocks. */
    fun finish() { put(ZEROS); out.flush() }

    private fun entry(path: String, size: Long, mtimeSec: Long, type: Char, mode: String) {
        require(path.isNotEmpty() && !path.startsWith("/")) { "tar path must be relative: $path" }
        val h = headed(path, size)
        if (h.pax != null) {
            put(header("PaxHeaders/" + asciiTail(path, 80), h.pax.size.toLong(), mtimeSec, 'x', "0000644", ""))
            put(h.pax); pad(h.pax.size.toLong())
        }
        put(header(h.name, if (size >= MAX_OCTAL_11) 0 else size, mtimeSec, type, mode, h.prefix))
    }

    private fun header(name: String, size: Long, mtimeSec: Long, type: Char, mode: String, prefix: String): ByteArray {
        val h = ByteArray(BLOCK)
        fun field(off: Int, len: Int, s: String) = s.toByteArray(Charsets.US_ASCII).copyInto(h, off, 0, minOf(len, s.length))
        field(0, 100, name)
        field(100, 8, "$mode\u0000")
        field(108, 8, "0000000\u0000")
        field(116, 8, "0000000\u0000")
        field(124, 12, java.lang.Long.toOctalString(size).padStart(11, '0') + "\u0000")
        field(136, 12, java.lang.Long.toOctalString(mtimeSec.coerceIn(0, MAX_OCTAL_11 - 1)).padStart(11, '0') + "\u0000")
        field(148, 8, "        ")
        h[156] = type.code.toByte()
        field(257, 8, "ustar\u000000")
        field(345, 155, prefix)
        val sum = h.sumOf { it.toInt() and 255 }
        field(148, 8, Integer.toOctalString(sum).padStart(6, '0') + "\u0000 ")
        return h
    }
}

private const val MAX_OCTAL_11 = 1L shl 33   // ai: 11 octal digits: sizes and times up to 8 GiB - 1 and 2242

// ai: How an entry is headed: its ustar name and prefix, and the pax record's body where a path fits neither field or is
// ai: not plain ASCII, or the size is past ustar's 8 GiB. The writer and the archive's size (Bundle) both read it.
private class Headed(val name: String, val prefix: String, val pax: ByteArray?)
private fun headed(path: String, size: Long): Headed {
    val ascii = path.toByteArray(Charsets.UTF_8).all { it in 0x20..0x7e }
    val split = if (ascii) splitUstar(path) else null
    val pax = StringBuilder()
    if (split == null) pax.append(paxRecord("path", path))
    if (size >= MAX_OCTAL_11) pax.append(paxRecord("size", size.toString()))
    return Headed(split?.second ?: asciiTail(path), split?.first ?: "", if (pax.isEmpty()) null else pax.toString().toByteArray(Charsets.UTF_8))
}
private fun padded(n: Long) = (n + BLOCK - 1) / BLOCK * BLOCK
private fun entryBytes(path: String, size: Long): Long = BLOCK + (headed(path, size).pax?.let { BLOCK + padded(it.size.toLong()) } ?: 0) + padded(size)

/** One item of a choice: a folder (no `open`), or a file of `size` bytes that `open` reads, at a relative path. */
class TarEntry(val path: String, val size: Long = 0, val mtimeSec: Long? = null, val open: (() -> InputStream)? = null) {
    val dir get() = open == null
}

/**
 * A choice of files and folders as one transfer, lizard-web/tar.mjs bundle's rule: one file alone is itself
 * (`single`); anything else one tar, named after its one folder or after its first item and how many more, each
 * file's folders added where missing, a path given twice numbered ("b (2).txt"), a folder no time was given for taking
 * the newest of what it holds, folders before what they hold, every entry in the order of its path.
 */
class Bundle(choice: List<TarEntry>) {
    val single: TarEntry?
    val name: String
    val entries: List<TarEntry>
    val files: Int
    /** The archive's bytes, or the one file's. */
    val size: Long

    init {
        require(choice.isNotEmpty()) { "nothing chosen" }
        val used = HashSet<String>()
        val all = choice.map { e ->
            if (e.dir) return@map e
            var p = e.path; var k = 2
            while (!used.add(p)) p = numbered(e.path, k++)
            if (p == e.path) e else TarEntry(p, e.size, e.mtimeSec, e.open)
        }
        val picked = all.filter { !it.dir }
        single = if (choice.size == 1 && picked.size == 1 && '/' !in picked[0].path) picked[0] else null
        files = picked.size
        val tops = LinkedHashSet<String>()
        for (e in all) tops.add(e.path.split('/').first { it.isNotEmpty() })
        name = single?.path ?: if (tops.size == 1) fitName(tops.first(), ".tar") else fitName(tops.first(), " and ${tops.size - 1} more.tar")
        val dirs = LinkedHashMap<String, Long?>()
        for (e in choice) {
            val parts = e.path.split('/').filter { it.isNotEmpty() }
            for (i in 1 until (if (e.dir) parts.size + 1 else parts.size)) {
                val p = parts.subList(0, i).joinToString("/")
                if (p !in dirs) dirs[p] = if (e.dir && i == parts.size) e.mtimeSec else null
            }
        }
        // ai: a folder with no time: the newest of what it holds, an empty one the newest of the choice (tar.mjs the same)
        val now = System.currentTimeMillis() / 1000
        val newest = picked.mapNotNull { it.mtimeSec }.maxOrNull() ?: now
        val folders = dirs.map { (p, t) ->
            TarEntry(p, mtimeSec = t ?: picked.filter { it.path.startsWith("$p/") }.mapNotNull { it.mtimeSec }.maxOrNull() ?: newest)
        }
        entries = if (single != null) listOf(single) else (folders + picked).sortedBy { it.path }
        size = single?.size ?: (entries.sumOf { entryBytes(if (it.dir) it.path.trimEnd('/') + "/" else it.path, it.size) } + ZEROS.size)
    }

    /** The archive into `out` (never called for a single file, which is sent as itself); its bytes. */
    fun write(out: OutputStream): Long {
        check(single == null) { "one file is sent as itself" }
        val w = TarWriter(out)
        val now = System.currentTimeMillis() / 1000
        for (e in entries) {
            if (e.dir) w.dir(e.path, e.mtimeSec ?: now)
            else e.open!!().use { w.file(e.path, e.size, e.mtimeSec ?: now, it) }
        }
        w.finish()
        return w.written
    }

    private companion object {
        fun numbered(path: String, k: Int): String {
            val slash = path.lastIndexOf('/') + 1; val dot = path.lastIndexOf('.')
            val cut = if (dot > slash) dot else path.length
            return "${path.substring(0, cut)} ($k)${path.substring(cut)}"
        }
        // ai: the transfer's header holds 255 bytes of UTF-8 of a name: the stem cut on a character so the suffix stays
        fun fitName(stem: String, suffix: String): String {
            var cps = stem.codePoints().toArray()
            while (cps.isNotEmpty() && (String(cps, 0, cps.size) + suffix).toByteArray(Charsets.UTF_8).size > 255) cps = cps.copyOf(cps.size - 1)
            return String(cps, 0, cps.size) + suffix
        }
    }
}

// ai: ustar's split of a plain ASCII path: the name field holds up to 100 bytes, the prefix up to 155 joined by a '/'.
private fun splitUstar(path: String): Pair<String, String>? {
    if (path.length <= 100) return "" to path
    var i = path.lastIndexOf('/', 155)
    while (i > 0) {
        val name = path.substring(i + 1)
        if (name.isNotEmpty() && name.length <= 100) return path.substring(0, i) to name
        i = path.lastIndexOf('/', i - 1)
    }
    return null
}

// ai: The ustar name a pax header stands in front of: the path's last characters as ASCII, for a reader with no pax.
private fun asciiTail(path: String, max: Int = 99): String =
    path.map { if (it.code in 0x20..0x7e) it else '_' }.joinToString("").takeLast(max).trimStart('/')

// ai: One pax record, "<length> <key>=<value>\n", its length counting its own digits.
private fun paxRecord(key: String, value: String): String {
    val rest = " $key=$value\n".toByteArray(Charsets.UTF_8).size
    var n = rest + 1
    while ((n.toString().length + rest) != n) n = n.toString().length + rest
    return "$n $key=$value\n"
}

/**
 * A tar, unpacked under `dest` (made if missing). Plain files and folders are made; pax (x, g) and GNU long-name (L)
 * headers are read for the path and size. Anything else (a link, a device, a FIFO), an absolute path or one through
 * "..", or a damaged header throws, leaving what was written so far for the caller to delete. Returns the files made,
 * in archive order.
 */
fun untar(src: InputStream, dest: File): List<File> {
    val made = ArrayList<File>()
    val root = dest.canonicalFile.also { it.mkdirs() }
    val h = ByteArray(BLOCK)
    var paxPath: String? = null; var paxSize: Long? = null; var longName: String? = null
    var zeros = 0
    while (true) {
        if (!readFully(src, h)) break
        if (h.all { it.toInt() == 0 }) { if (++zeros == 2) break else continue }
        zeros = 0
        if (!checksumOk(h)) throw IOException("a damaged tar header")
        val type = (h[156].toInt() and 255).toChar()
        val size = paxSize ?: octal(h, 124, 12)
        val name = cstr(h, 0, 100); val prefix = if (cstr(h, 257, 6) == "ustar") cstr(h, 345, 155) else ""
        val path = paxPath ?: longName ?: if (prefix.isEmpty()) name else "$prefix/$name"
        when (type) {
            'x', 'g' -> {
                val body = readBody(src, size)
                if (type == 'x') parsePax(body).let { paxPath = it["path"] ?: paxPath; paxSize = it["size"]?.toLongOrNull() ?: paxSize }
                continue
            }
            'L' -> { longName = String(readBody(src, size), Charsets.UTF_8).trimEnd('\u0000'); continue }
            '0', '\u0000', '7', '5' -> {
                val target = safeTarget(root, path)
                if (type == '5' || path.endsWith("/")) target.mkdirs()
                else {
                    target.parentFile?.mkdirs()
                    FileOutputStream(target).use { copyExactly(src, it, size) }
                    skipPad(src, size)
                    made.add(target)
                }
                if (type == '5') skipPad(src, size)
            }
            else -> throw IOException("tar entry '$type' at $path: only files and folders unpack")
        }
        paxPath = null; paxSize = null; longName = null
    }
    return made
}

// ai: An entry's place under root: relative, no "..", no empty parts; refused otherwise rather than cleaned up.
private fun safeTarget(root: File, path: String): File {
    val parts = path.removePrefix("./").split('/').filter { it.isNotEmpty() && it != "." }
    if (path.startsWith("/") || parts.isEmpty() || parts.any { it == ".." } || (parts[0].length == 2 && parts[0][1] == ':'))
        throw IOException("tar path refused: $path")
    val f = File(root, parts.joinToString(File.separator)).canonicalFile
    if (!f.path.startsWith(root.path + File.separator)) throw IOException("tar path refused: $path")
    return f
}

private fun readFully(src: InputStream, b: ByteArray): Boolean {
    var off = 0
    while (off < b.size) { val n = src.read(b, off, b.size - off); if (n < 0) { if (off == 0) return false; throw IOException("a tar cut short") }; off += n }
    return true
}

private fun readBody(src: InputStream, size: Long): ByteArray {
    if (size > (1 shl 20)) throw IOException("a tar extended header of $size bytes")
    val b = ByteArray(size.toInt()); if (size > 0 && !readFully(src, b)) throw IOException("a tar cut short")
    skipPad(src, size); return b
}

private fun copyExactly(src: InputStream, out: OutputStream, size: Long) {
    val buf = ByteArray(1 shl 20); var left = size
    while (left > 0) { val n = src.read(buf, 0, minOf(buf.size.toLong(), left).toInt()); if (n < 0) throw IOException("a tar cut short"); out.write(buf, 0, n); left -= n }
}

private fun skipPad(src: InputStream, size: Long) {
    var p = ((BLOCK - size % BLOCK) % BLOCK).toInt()
    val buf = ByteArray(BLOCK)
    while (p > 0) { val n = src.read(buf, 0, p); if (n < 0) throw IOException("a tar cut short"); p -= n }
}

private fun parsePax(body: ByteArray): Map<String, String> {
    val out = HashMap<String, String>(); var i = 0
    while (i < body.size) {
        val sp = (i until body.size).firstOrNull { body[it] == ' '.code.toByte() } ?: break
        val len = String(body, i, sp - i, Charsets.US_ASCII).toIntOrNull() ?: break
        if (len <= 0 || i + len > body.size) break
        val rec = String(body, sp + 1, i + len - sp - 2, Charsets.UTF_8)
        val eq = rec.indexOf('='); if (eq > 0) out[rec.substring(0, eq)] = rec.substring(eq + 1)
        i += len
    }
    return out
}

private fun cstr(h: ByteArray, off: Int, len: Int): String {
    var end = off; while (end < off + len && h[end].toInt() != 0) end++
    return String(h, off, end - off, Charsets.UTF_8)
}

private fun octal(h: ByteArray, off: Int, len: Int): Long {
    if (h[off].toInt() and 0x80 != 0) {   // ai: GNU's base-256 form for sizes past 8 GiB
        var v = 0L; for (i in off + 1 until off + len) v = (v shl 8) or (h[i].toLong() and 255); return v
    }
    return cstr(h, off, len).trim().ifEmpty { "0" }.toLong(8)
}

private fun checksumOk(h: ByteArray): Boolean {
    var sum = 0L
    for (i in h.indices) sum += if (i in 148 until 156) 32 else (h[i].toInt() and 255)
    return cstr(h, 148, 8).trim().toLongOrNull(8) == sum
}
