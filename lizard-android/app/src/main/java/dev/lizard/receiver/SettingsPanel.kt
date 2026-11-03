package dev.lizard.receiver

import dev.lizard.pick.Pick
import dev.lizard.readout.Readout
import java.util.Locale
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.doubleOrNull
import dev.lizard.ui.*

// ai: Settings, as the web's pages hold them since 2026-10-02. Receive: under the transfer, Settings (the decoder and
// ai: its batch size, the camera: lens, resolution, format, zoom, crop), Developer Tools (Save replays, the lab line, the
// ai: readout: the decoder, camera, code, file, lock and heat a line each) and About,
// ai: the camera running beside them. Home: the received files with Delete all, About. The Settings screen, reached by a gear on
// ai: Home and Receive, went with them. Options bare, no descriptions.

// ai: The readout, a labelled line or two a part, from the session's stats JSON (the receiver's, its camera's report:
// ai: what was asked and given, the newest capture), the phase lock's state and the heat; a part with nothing to say
// ai: is left out.
internal fun MainActivity.devReadout(): String {
    val j = stats
    val f = { x: Double, d: Int -> String.format(Locale.ROOT, "%.${d}f", x) }
    val out = mutableListOf<String>()
    fun put(label: String, vararg lines: String?) {
        lines.filterNotNull().filter { it.isNotEmpty() }.forEachIndexed { i, l -> out.add((if (i == 0) label else "").padEnd(9) + l) }
    }
    // ai: a whole figure (the stats' counts)
    fun JsonObject.int(k: String) = num(k).toLong()
    if (j != null) {
        val dec = j.str("decoder")
        put("decoder", if (dec.startsWith("gpu")) "GPU ${dec.removePrefix("gpu").trim()}${if (j.flag("zeroCopy")) ", zero-copy" else ""}, " +
                "batches up to ${j.int("cap")}, ${f(j.num("gpuMs"), 2)} ms of GPU a frame"
            else if (dec == "cpu") "CPU, ${j.int("threads")} of ${j.int("threadsMax")} threads${if (j.flag("simd")) ", NEON" else ""}, " +
                "${f(j.num("cpuMs"), 1)} ms a frame"
            // ai: PC (liblizard/core/rx/stream.h): the stream's own line, the PC's decode after it
            else if (dec.startsWith("pc")) j.obj("stream")?.let { st ->
                (if (st.flag("connected")) "to the PC at ${st.str("peer")}" else "waiting for the PC on port ${st.int("port")}") +
                    ", ${f(st.num("sentFps"), 0)} of ${f(st.num("cameraFps"), 0)} frames a second sent, ${f(st.num("MBps"), 1)} MB/s, " +
                    "${st.int("dropped")} dropped${dec.removePrefix("pc").removePrefix(":").trim().let { if (it.isEmpty()) "" else "; the PC's $it" }}"
            } ?: dec
            else dec,
            j.str("gpuWhy").takeIf { dec == "cpu" && it.isNotEmpty() }?.let { "no GPU: $it" })
    }
    val c = cam
    val now = j?.obj("camera")?.obj("report")?.takeIf { it.num("exposureMs") > 0 }
    put("camera", c?.let { "${it.id}: ${it.size.width}x${it.size.height} ${it.format} at ${it.fps} fps, ${settings.layout} crop${if (it.note.isEmpty()) "" else ", ${it.note}"}" } ?: "not open",
        j?.let { "${f(it.num("capturedFps"), 0)} captured, ${f(it.num("processedFps"), 0)} decoded, ${it.int("dropped")} dropped a second" +
            if ("heldMs" in it) ", held ${f(it.num("heldMs"), 0)} ms (${f(it.num("heldMaxMs"), 0)} max)" else "" },
        now?.let { "exposure ${f(it.num("exposureMs"), 2)} ms, ISO ${it.int("iso")}, readout ${f(it.num("readoutMs"), 2)} ms" +
            (if (it["lens"] == null || it["lens"] is JsonNull) "" else ", lens ${it.str("lens")}") })
    if (j != null) {
        val word = j.obj("word")
        val v = word?.int("version")?.toInt() ?: 0
        put("code", "found ${f(100 * j.num("foundShare"), 0)}%" + (if (j.num("side") > 0) ", symbol ${f(j.num("side"), 0)} px" else "") +
                (if (v > 0) ", LIZARD-${Pick.STEP * v}${word?.int("fps")?.takeIf { it > 0 }?.let { " at $it fps" } ?: ""}" else ", no word yet"),
            "${Readout.rate(j.num("goodputKBs"))}, ${j.int("blocks")} blocks in ${f(j.num("windowSecs"), 2)} s" +
                (j.obj("totals")?.let { t -> "; total ${t.int("blocks")} blocks in ${t.int("frames")} frames" } ?: ""))
        val file = j.obj("file")
        put("file", when {
            j.flag("test") -> "the test stream"
            file == null -> "none yet"
            else -> "${file.str("name").ifEmpty { "(no name)" }}: ${Readout.partOf(file.num("received"), file.int("size"))}" +
                ", chunks ${file.int("chunksVerified")} of ${file.int("chunks")} verified${if (file.flag("verified")) ", whole" else ""}"
        })
    }
    // ai: the phase lock (the session's, lock.h: PhaseLock.State's fields; NaN as null), or why it is off
    receive.lock?.let { p ->
        val real = { k: String -> (p[k] as? JsonPrimitive)?.doubleOrNull ?: Double.NaN }
        if ("off" in p) put("lock", "off: ${p.str("off")}")
        else if ("arm" in p) {
            val pct = { x: Double -> if (x.isNaN()) "-" else "${f(100 * x, 0)}%" }
            put("lock", "${p.str("arm")} ${p.str("state")}: ${pct(real("a"))} before, ${pct(real("b"))} after; gain ${f(p.num("gain"), 2)}, " +
                "pace ${f(p.num("pace"), 0)} us a second, ${p.int("delays")} delays")
        }
    }
    put("heat", "thermal $heat" + (clocks?.let { ", GPU ${it.mhz} of ${it.top} MHz" } ?: "") + (powerW?.let { ", ${f(it, 1)} W" } ?: "") +
        (if (batteryC > 0) ", battery ${f(batteryC, 1)} C" else ""))
    j?.str("error")?.takeIf { it.isNotEmpty() }?.let { put("error", it) }
    return out.joinToString("\n")
}

internal fun MainActivity.aboutLine() = "LIZARD ${version()}${installed().let { if (it.isEmpty()) "" else ", installed $it" }}"
