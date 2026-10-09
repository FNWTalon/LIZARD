package dev.lizard.receiver

import java.util.Locale
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt

// ai: The receive screen's words and figures, the web receiver's (lizard-web/recv.mjs showState, ui.mjs bytes, rate,
// ai: left) in Kotlin: one wording on every surface (2026-10-01). The user's figures are progress, size, speed and time
// ai: left; the format's name and the registered
// ai: share are the lab line's, in Developer Tools. 1 KB = 1000 B, a figure kept on its line (a no-break space), digits as the
// ai: web prints them whatever the phone's locale. Pure: no Android in here.
object Readout {
    private const val NB = ' '

    // ai: "812 B", "7.4 MB", "31 MB": the unit chosen after rounding, so 999,500 B reads "1.0 MB" (ui.mjs bytes)
    fun bytes(n: Long): String {
        if (n < 1000) return "$n${NB}B"
        val (v, u) = when { n < 999_500 -> n / 1e3 to "KB"; n < 999_500_000 -> n / 1e6 to "MB"; else -> n / 1e9 to "GB" }
        return "${at(v)}$NB$u"
    }
    private fun at(v: Double) = if (v < 9.95) String.format(Locale.ROOT, "%.1f", v) else v.roundToInt().toString()

    // ai: "6.3 of 12.9 MB": the part in the unit bytes() gives the whole (recv.mjs partOf)
    fun partOf(n: Double, whole: Long): String {
        val all = bytes(whole)
        val unit = all.substringAfter(NB)
        val v = n / when (unit) { "KB" -> 1e3; "MB" -> 1e6; "GB" -> 1e9; else -> 1.0 }
        return "${if (unit == "B") v.roundToInt().toString() else at(v)} of $all"
    }

    // ai: "640 KB/s", "1.57 MB/s" (ui.mjs rate)
    fun rate(kbs: Double): String = if (kbs < 999.5) "${kbs.roundToInt()}${NB}KB/s" else "${String.format(Locale.ROOT, "%.2f", kbs / 1000)}${NB}MB/s"

    // ai: "4 s left", "2 min left" from 100 s (ui.mjs left)
    fun left(secs: Double): String {
        val s = maxOf(1, ceil(secs).toInt())
        return if (s < 100) "$s${NB}s left" else "${(s / 60.0).roundToInt()}${NB}min left"
    }

    enum class Tone { Plain, Good, Bad }

    // ai: line: the state; nums: the figures under it ("" for none); frac: the meter (null hides it)
    data class Line(val text: String, val tone: Tone = Tone.Plain, val nums: String = "", val frac: Double? = null)

    // ai: What the native receiver says (its stats JSON, receiver_stub.cpp's header) that the screen reads.
    data class Rx(
        val state: String = "", val error: String = "", val goodputKBs: Double = 0.0, val foundShare: Double = 0.0,
        val bandVersion: Int = 0, val fps: Int = 0, val hasFile: Boolean = false, val name: String = "a file",
        val size: Long = 0, val received: Long = 0, val verified: Boolean = false,
        // ai: the file's bytes as sent (every chunk's zstd frame or its own; 0 until the manifest has said them) and
        // ai: how many are in: what the light carries, which the rate and the time left count (2026-10-05)
        val sent: Long = 0, val sentIn: Long = 0,
    )

    // ai: One state at a time, the first that holds, as recv.mjs showState: error, starting, getting ready (no answer
    // ai: from the receiver yet), receiving, received, camera off, the test stream, or looking (a word read with no file
    // ai: yet says "Looking" too since 2026-10-05: the receiver's "found" is its held word, which stays once read, so
    // ai: "Found the code, waiting for the file" stood after the code had left the view). on: the camera runs; ready: the receiver has planned (its state past "starting"); secs: the
    // ai: transfer's own clock, for the received line.
    // ai: no answer from the receiver yet: the line's "Getting ready", and the preview's cover with its spinner (Receive.kt)
    fun gettingReady(on: Boolean, starting: Boolean, loading: Boolean, rx: Rx) =
        starting || loading || (on && (rx.state.isEmpty() || rx.state == "starting" || rx.state == "loading"))
    // ai: how much of the file is in: the bytes as sent where the manifest has said them, the file's own until then; the
    // ai: meter, the figures' percentage and the rail's (Parts.kt PctSquare)
    fun fraction(rx: Rx): Double {
        val total = if (rx.sent > 0) rx.sent else rx.size
        val got = if (rx.sent > 0) rx.sentIn else rx.received
        return if (total > 0) (got.toDouble() / total).coerceIn(0.0, 1.0) else 0.0
    }
    fun line(on: Boolean, starting: Boolean, loading: Boolean, error: String?, rx: Rx, secs: Double): Line = when {
        error != null -> Line(error, Tone.Bad)
        starting -> Line("Starting the camera")
        gettingReady(on, false, loading, rx) -> Line("Getting ready")
        rx.state == "error" -> Line("Reading stopped. Start the camera again to go on.", Tone.Bad)
        // ai: the test stream first while the camera reads it (2026-10-05): it takes over from a file received or in
        // ai: progress, which shows again once the camera is back on its frames (receiver.cpp's state the same)
        on && rx.state == "test" -> Line("Reading the test stream")
        rx.hasFile && !rx.verified && on -> {
            // ai: the bytes as sent where the manifest has said them, the file's own until then (recv.mjs the same)
            val total = if (rx.sent > 0) rx.sent else rx.size
            val got = if (rx.sent > 0) rx.sentIn else rx.received
            val frac = fraction(rx)
            val now = rx.goodputKBs
            // ai: the rate itself is its own line under the buttons since 2026-10-05 (Receive.kt RateLine), so not here
            val leftNow = if (now > 0) ", ${left((total - got) / (now * 1000))}" else ""
            Line("Receiving ${rx.name}", nums = "${floor(100 * frac).toInt()}%, ${partOf(got.toDouble(), total)}$leftNow", frac = frac)
        }
        rx.hasFile && rx.verified -> Line("Received ${rx.name}, ${bytes(rx.size)} in ${String.format(Locale.ROOT, "%.1f", secs)}${NB}s${if (rx.sent in 1 until rx.size) ", ${bytes(rx.sent)} sent" else ""}", Tone.Good, frac = 1.0)
        // ai: nothing while the camera is off (2026-10-02); the screen draws no empty line
        !on -> Line("")
        // ai: nothing while looking (2026-10-05: "Looking for a code" until then; the rate under the buttons says it)
        else -> Line("")
    }

    // ai: The lab line (Developer Tools): the last second's rate, the registered share and, while a word is read, the format,
    // ai: the web's #lab ("1571 KB/s, 97% registered, LIZARD-480 at 60 fps", or "no code yet").
    fun lab(on: Boolean, rx: Rx): String {
        if (!on) return ""
        if (rx.foundShare <= 0 && rx.goodputKBs <= 0) return "no code yet"
        val named = if (rx.bandVersion > 0 && rx.state != "looking") ", LIZARD-${8 * rx.bandVersion}${if (rx.fps > 0) " at ${rx.fps}${NB}fps" else ""}" else ""
        return "${rx.goodputKBs.roundToInt()}${NB}KB/s, ${(100 * rx.foundShare).roundToInt()}% registered$named"
    }
}
