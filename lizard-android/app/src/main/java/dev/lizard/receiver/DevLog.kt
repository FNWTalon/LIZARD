package dev.lizard.receiver

import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

// ai: The rig's development log (lizard-web/server.mjs /api/stats): the stats JSON posted once a second, best
// ai: effort and never awaited, as the web's devlog.mjs logPost. Nothing read back. A post still in flight skips the
// ai: next; a failed one rests the path REST_MS.
class DevLog {
    private val pool = Executors.newSingleThreadExecutor { r -> Thread(r, "devlog").apply { isDaemon = true } }
    private val busy = AtomicBoolean(false)
    @Volatile private var restUntil = 0L

    fun post(address: String, path: String, body: String) {
        if (address.isBlank() || System.currentTimeMillis() < restUntil || !busy.compareAndSet(false, true)) return
        val base = address.trim().trimEnd('/').let { if (it.contains("://")) it else "http://$it" }
        pool.execute {
            try {
                val c = URL(base + path).openConnection() as HttpURLConnection
                c.connectTimeout = 1000; c.readTimeout = 1000
                c.requestMethod = "POST"; c.doOutput = true
                c.setRequestProperty("content-type", "application/json")
                c.outputStream.use { it.write(body.toByteArray()) }
                if (c.responseCode != 200) throw java.io.IOException("HTTP ${c.responseCode}")
                c.inputStream.use { it.readBytes() }
                c.disconnect()
            } catch (_: Exception) {
                restUntil = System.currentTimeMillis() + REST_MS
            } finally { busy.set(false) }
        }
    }

    companion object { const val REST_MS = 10_000L }
}
