package dev.lizard.desktop

import dev.lizard.phase.PhaseLock
import kotlin.math.roundToLong

// ai: The phase lock's driver on a PC (2026-10-25): the Android app's PhaseLock (liblizard/kotlin/PhaseLock.kt, used
// ai: as it is) steering a camera that takes a software trigger (rxCamera's "trigger"), the session firing it on a
// ai: schedule of its own (camera/trigger.h) where Camera2 lengthens one frame. Only the simulated camera (trigger=1)
// ai: takes a software trigger so far (a UVC camera has none: the lock is off for it, its reason in the readout).
// ai: Begun once the stream runs: the trigger every P = 1e9 / f * (1 - SLACK) ns for the painted rate f (60 until a
// ai: word names one, the stats' word.fps), the camera a little faster than the display as the S26's 16.650 ms
// ai: against 16.667, so the lock's corrections are small delays later; then a thread ("lizard-lock") takes each
// ai: capture as the session hands it over (rxCaptures: its trigger's deadline, the schedule's exact time without the
// ai: host's lateness in sending it (camera/trigger.h), and whether a delay moved it; the running
// ai: schedule's own captures alone, so the lock's first interval after a begin is that schedule's period, never one
// ai: from a capture of the free run or of the schedule before still in flight), feeds the lock the receiver's newest
// ai: frames first (rxSeries, in scan and track only, as the app's Engine.feed) and moves the schedule by what it asks
// ai: (rxDelay). A word naming another rate: P again (rxTrigger) and a new PhaseLock, its clock and period assumptions
// ai: afresh; otherwise the lock outlives a stop (reopened, as the app keeps one an engine). Ended at a stop, before the
// ai: camera's own stop, the trigger put back to free run. LIZ_PHASE: off, track (unset), scan, or a hold at that many
// ai: us (the app's debug.lizard.phase); off leaves the camera free-running. The camera's exposure is the user's
// ai: setting; the lock sets none. Its lines go to stderr as "lock: ...". start and end are the session's thread's
// ai: (ReceiveState's worker) alone; state, why and wordFps from any.
class Lock {
    companion object {
        const val SLACK = 1e-3
        // ai: the trigger's period for a painted rate: 16,650,000 ns at 60
        fun periodNs(fps: Int) = (1e9 / fps * (1 - SLACK)).roundToLong()
        private val asked = System.getenv("LIZ_PHASE")?.trim()?.lowercase().orEmpty().ifEmpty { "track" }
        private fun say(line: String) = System.err.println("lock: $line")
    }

    // ai: why the lock is not running (Developer Tools' "off: <why>"); "" while it runs
    @Volatile var why = "the camera is off"
        private set
    // ai: the painted rate the newest word names (ReceiveState's poll), 0 none yet
    @Volatile var wordFps = 0
    @Volatile private var lock: PhaseLock? = null
    val state: PhaseLock.State? get() = if (thread != null) lock?.state else null

    private var lockFps = 0                 // ai: the rate lock was made for
    @Volatile private var thread: Thread? = null
    @Volatile private var running = false
    private var h = 0L
    private var triggered = false

    // ai: the mode LIZ_PHASE asks; null: off
    private fun modeOf(v: String): Pair<PhaseLock.Mode, Long>? = when (v) {
        "track" -> PhaseLock.Mode.Track to 0L
        "scan" -> PhaseLock.Mode.Scan to 0L
        else -> v.toLongOrNull()?.let { PhaseLock.Mode.Hold to it }
    }

    // ai: once the stream runs (the session's thread): the trigger and the thread, where the camera takes a trigger
    fun start(h: Long, trigger: Boolean) {
        end()
        if (!trigger) { why = "the camera takes no trigger"; return }
        val mode = modeOf(asked) ?: run { why = "LIZ_PHASE=$asked"; return }
        val fps = wordFps.takeIf { it in 1..240 } ?: lockFps.takeIf { it > 0 } ?: 60
        val err = runCatching { Native.rxTrigger(h, periodNs(fps)) }.getOrElse { it.message ?: it.javaClass.simpleName }
        if (err.isNotEmpty()) { why = "the trigger refused: $err"; say(why); return }
        this.h = h
        triggered = true
        running = true
        why = ""
        say("the trigger every %.3f ms (%d fps painted)".format(java.util.Locale.ROOT, periodNs(fps) / 1e6, fps))
        thread = Thread({ loop(h, fps, mode) }, "lizard-lock").apply { isDaemon = true; start() }
    }

    // ai: the thread ended (rxWake ends its wait), then the trigger put back to free run; the session's thread
    fun end() {
        val t = thread
        if (t != null) {
            running = false
            runCatching { Native.rxWake(h) }
            t.join(2000)
            thread = null
        }
        if (triggered) { runCatching { Native.rxTrigger(h, 0) }; triggered = false }
        why = "the camera is off"
    }

    private fun made(fps: Int, mode: Pair<PhaseLock.Mode, Long>): PhaseLock =
        PhaseLock({ say(it) }, gridNs = (1e9 / fps).roundToLong()).also { it.set(mode.first, mode.second); lock = it; lockFps = fps }

    private fun loop(h: Long, fps0: Int, mode: Pair<PhaseLock.Mode, Long>) {
        var fps = fps0
        var p = lock?.takeIf { lockFps == fps }?.also { it.reopened() } ?: made(fps, mode)
        var refused = 0     // ai: a rate the trigger refused, not asked again until the word names another
        var fedMs = -1.0
        var n = 0
        try {
            while (running) {
                val f = wordFps
                if (f in 1..240 && f != fps && f != refused) {
                    val err = Native.rxTrigger(h, periodNs(f))
                    if (err.isEmpty()) {
                        say("%d fps painted: the trigger every %.3f ms, the lock afresh".format(java.util.Locale.ROOT, f, periodNs(f) / 1e6))
                        fps = f; p = made(f, mode); fedMs = -1.0
                    } else { refused = f; say("the trigger refused $f fps: $err") }
                }
                val c = Native.rxCaptures(h, 200)
                var i = 0
                while (i + 1 < c.size && running) {
                    val ts = c[i]
                    fedMs = feed(h, p, fedMs)
                    val us = p.onCapture(ts, c[i + 1] != 0L)
                    if (us > 0) Native.rxDelay(h, us * 1000)
                    if (++n % 300 == 0) p.line(ts)?.let { say(it) }
                    i += 2
                }
            }
        } catch (e: Throwable) {
            why = "the lock stopped: ${e.message ?: e.javaClass.simpleName}"
            say(why)
        }
    }

    // ai: the receiver's frames since the newest fed (rxSeries: the held word's version, then a frame's ms on the
    // ai: capture clock, verified and new blocks, registered, the pilots' readings and errors, even blocks then odd) to
    // ai: the lock, in scan and track only; the newest frame's ms
    private fun feed(h: Long, p: PhaseLock, since: Double): Double {
        if (p.mode != PhaseLock.Mode.Scan && p.mode != PhaseLock.Mode.Track) return since
        val a = Native.rxSeries(h, since)
        val n = (a.size - 1) / 8
        if (n <= 0) return since
        val frames = ArrayList<PhaseLock.Frame>(n)
        for (i in 0 until n) {
            val o = 1 + 8 * i
            frames.add(PhaseLock.Frame((a[o] * 1e6).toLong(), a[o + 1].toInt(), a[o + 3] == 1.0, a[o + 4], a[o + 5], a[o + 6], a[o + 7]))
        }
        p.onWindow(frames, a[0].toInt())
        return a[1 + 8 * (n - 1)]
    }
}
