package dev.lizard.desktop

import java.io.File
import java.net.StandardProtocolFamily
import java.net.UnixDomainSocketAddress
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.channels.ServerSocketChannel
import java.nio.channels.SocketChannel
import java.nio.file.Files
import java.nio.file.Path
import kotlin.concurrent.thread

// ai: One sender takes what is handed in (2026-10-23): paths given to a later launch (a terminal's arguments, a file
// ai: manager's Open with) go to the sender already running, over a Unix socket in the session's runtime folder, and
// ai: that launch ends; the first sender listens there. A socket left by a sender that died is taken over. A launch with
// ai: no paths while one listens opens a window of its own, as before.
object Instance {
    private val path: Path = Path.of(System.getenv("XDG_RUNTIME_DIR")?.takeIf { it.isNotEmpty() } ?: System.getProperty("java.io.tmpdir"), "lizard-sender.sock")

    private fun connect(): SocketChannel? = runCatching {
        SocketChannel.open(StandardProtocolFamily.UNIX).apply { connect(UnixDomainSocketAddress.of(path)) }
    }.getOrNull()

    // ai: the paths to the sender listening, one a line; false where none listens
    fun handOff(files: List<File>): Boolean {
        val ch = connect() ?: return false
        return runCatching {
            ch.use { val b = ByteBuffer.wrap(files.joinToString("\n") { it.path }.toByteArray()); while (b.hasRemaining()) it.write(b) }
            true
        }.getOrDefault(false)
    }

    // ai: listen for later launches' paths, unless a sender already does; take gets those that exist, on this thread
    fun listen(take: (List<File>) -> Unit) {
        connect()?.let { it.close(); return }
        val server = runCatching {
            Files.deleteIfExists(path)
            ServerSocketChannel.open(StandardProtocolFamily.UNIX).apply { bind(UnixDomainSocketAddress.of(path)) }
        }.getOrElse { System.err.println("instance: not listening: ${it.message}"); return }
        Runtime.getRuntime().addShutdownHook(Thread { runCatching { server.close(); Files.deleteIfExists(path) } })
        thread(name = "lizard-instance", isDaemon = true) {
            while (true) {
                val c = runCatching { server.accept() }.getOrNull() ?: break
                runCatching {
                    val text = c.use { Channels.newInputStream(it).readBytes().decodeToString() }
                    val files = text.lines().filter { it.isNotEmpty() }.map(::File).filter { it.exists() }
                    if (files.isNotEmpty()) take(files)
                }
            }
        }
    }
}
