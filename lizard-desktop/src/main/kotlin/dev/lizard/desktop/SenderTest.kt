package dev.lizard.desktop

import kotlinx.coroutines.delay
import kotlin.system.exitProcess

// ai: The integration check's hook (2026-10-04, the desktop sender's build): env LIZ_SENDER_TEST, space separated
// ai: key=value, e.g. "secs=8 codes=2 enc=cpu blocks=0". The payload is the test stream; every setting starts at its
// ai: default (the kept ones neither read nor written: Prefs.forget), the ones named applied; sending starts once the
// ai: code area has a size, and after secs (8 unless named) of presenting the last presenter and sender stats are
// ai: printed to stdout as one JSON line {"present": {...}, "sender": {...}} and the app exits 0. Any failure prints
// ai: "FAIL <why>" and exits 1. Keys: secs; codes 1 or 2; enc auto, gpu or cpu; blocks 0 (auto) to 128; fps 1 to 60;
// ai: size 25 to 100 (%); spacing 4 to 68 and margin 2 to 34 (modules); ring auto, 32, 64, 128 or 256; full 0 or 1 (the window manager's full
// ai: screen); vsync 0 or 1 (1 unless named); file=<path>[:<path>...] (files or folders in place of the test stream, as a pick or a drop
// ai: gives them: one file as itself, else one tar, 2026-10-16; the paths split on the platform's separator, ';' on
// ai: Windows); steps
// ai: (2026-10-04, the app's own controls, which only a hand had moved): comma separated <seconds>:<action>, from when
// ai: the presenter first presents, each as the control does it: full (Fullscreen, F11), max (the window maximized), pause, resume, stop, start,
// ai: size=N, spacing=N, margin=N, codes=N, blocks=N (0 auto) (each a slider's release), fps=N (any 1 to 60, a chip's or the slider's), vsync=0|1,
// ai: enc=auto|gpu|cpu, rail, press (Start as
// ai: the button presses it: the brightness tip first where it is due), confirm and cancel (the tip's buttons), and the
// ai: selection's (2026-10-23): add=<path>[:<path>...] (as a pick adds; ';' on Windows), remove=<an item's name> (its Remove), clear (Clear),
// ai: list (the items and the state line on stderr as "list: <name> | ... | <state line>"); a change stops the run, as
// ai: the app's does, until a start; each logged on stderr as "step <s>: <action>" when it is taken. Each second's
// ai: presenter stats go to stderr as "present: {...}".
class SenderTest private constructor(private val secs: Double, private val kv: Map<String, String>) {
    companion object {
        private val KEYS = setOf("secs", "codes", "enc", "blocks", "fps", "vsync", "size", "spacing", "margin", "ring", "full", "file", "steps")
        // ai: how long the presenter may take to start (the canvas shown, the GPU painter made) before the check fails
        private const val START_SECS = 60.0

        fun parse(env: String?): SenderTest? {
            if (env == null) return null
            val kv = env.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.associate { w ->
                val i = w.indexOf('=')
                if (i <= 0) fail("LIZ_SENDER_TEST: \"$w\" is not key=value")
                w.substring(0, i) to w.substring(i + 1)
            }
            (kv.keys - KEYS).let { if (it.isNotEmpty()) fail("LIZ_SENDER_TEST: unknown ${it.joinToString()}") }
            val secs = kv["secs"]?.let { it.toDoubleOrNull()?.takeIf { s -> s > 0 } ?: fail("LIZ_SENDER_TEST: secs=$it") } ?: 8.0
            return SenderTest(secs, kv)
        }

        // ai: the run stopped first (the presenter's thread and the painters joined), so a FAIL is exit 1 and never a
        // ai: driver's teardown under a present in flight (2026-10-04); stop is the UI thread's, asked there
        var running: SendState? = null
        fun fail(why: String): Nothing {
            println("FAIL $why")
            System.out.flush()
            running?.let { s ->
                runCatching {
                    if (java.awt.EventQueue.isDispatchThread()) s.stop() else java.awt.EventQueue.invokeAndWait { s.stop() }
                }
            }
            exitProcess(1)
        }
    }

    private fun int(k: String, range: IntRange): Int? = kv[k]?.let { v -> v.toIntOrNull()?.takeIf { it in range } ?: fail("LIZ_SENDER_TEST: $k=$v") }

    // ai: the settings named
    fun apply(s: SendState) {
        s.test = true
        kv["file"]?.let { paths ->
            val fs = paths.split(java.io.File.pathSeparatorChar).filter { it.isNotEmpty() }.map { java.io.File(it) }
            fs.firstOrNull { !it.exists() }?.let { fail("LIZ_SENDER_TEST: file=$it is not there") }
            if (fs.isEmpty()) fail("LIZ_SENDER_TEST: file= names nothing")
            s.test = false
            s.pick(fs)
        }
        int("codes", 1..2)?.let { s.codes = it }
        kv["enc"]?.let { if (it in setOf("auto", "gpu", "cpu")) s.enc = it else fail("LIZ_SENDER_TEST: enc=$it") }
        int("blocks", 0..128)?.let { s.blocks = it }
        int("fps", 1..60)?.let { s.fps = it }
        int("vsync", 0..1)?.let { s.vsync = it == 1 }
        int("size", 25..100)?.let { s.size = it }
        int("spacing", Pick.SPACING_MIN..Pick.SPACING_MAX)?.let { s.spacing = it }
        int("margin", Pick.QUIET..Pick.MARGIN_MAX)?.let { s.margin = it }
        kv["ring"]?.let { v -> s.ring = if (v == "auto") -1 else Pick.RINGS.indexOf(v.toIntOrNull() ?: 0).takeIf { it >= 0 } ?: fail("LIZ_SENDER_TEST: ring=$v") }
    }

    // ai: on the UI thread (Main.kt's LaunchedEffect)
    suspend fun run(s: SendState) {
        running = s
        if (s.loadError.isNotEmpty()) fail(s.loadError)
        if (int("full", 0..1) == 1) s.toggleFull()
        // ai: a pick is walked off the UI thread (SendState.pick): Start once it is in
        val t0 = System.nanoTime()
        while (s.reading) { if (System.nanoTime() - t0 > START_SECS * 1e9) fail("the files were not read in ${START_SECS.toInt()} s"); delay(50) }
        (s.phase as? SendState.Phase.Error)?.let { fail(it.why) }
        s.start()
        while (!s.presenting) {
            (s.phase as? SendState.Phase.Error)?.let { fail(it.why) }
            if (s.phase == SendState.Phase.Idle) fail("the sender did not start")
            if (System.nanoTime() - t0 > START_SECS * 1e9) fail("no presenter after ${START_SECS.toInt()} s (${s.phase}, the area ${s.areaW} x ${s.areaH} device px, the canvas ${if (s.canvas.isShowing) "shown" else "not shown"})")
            delay(50)
        }
        val begun = System.nanoTime()
        val end = begun + (secs * 1e9).toLong()
        val todo = steps().toMutableList()
        var stopped = false
        var seen = ""
        while (System.nanoTime() < end) {
            (s.phase as? SendState.Phase.Error)?.let { fail(it.why) }
            val at = (System.nanoTime() - begun) / 1e9
            while (todo.isNotEmpty() && todo.first().first <= at) {
                val (t, act) = todo.removeAt(0)
                System.err.println("step %.1f: %s".format(java.util.Locale.ROOT, t, act))
                stopped = step(s, act, stopped)
            }
            if (stopped && s.presenting) stopped = false
            if (!s.presenting && !stopped && s.phase != SendState.Phase.Preparing) fail("the presenter stopped")
            // ai: each second's presenter stats as they come (the poll's), on stderr: a cadence is read from all of them
            if (s.presentRaw.isNotEmpty() && s.presentRaw !== seen) { seen = s.presentRaw; System.err.println("present: $seen") }
            delay(100)
        }
        if (s.presentRaw.isEmpty()) s.poll()
        (s.phase as? SendState.Phase.Error)?.let { fail(it.why) }
        val line = "{\"present\": ${s.presentRaw.ifEmpty { "{}" }}, \"sender\": ${s.senderRaw.ifEmpty { "{}" }}}"
        s.stop()
        println(line)
        System.out.flush()
        exitProcess(0)
    }

    // ai: the steps as (seconds, action), in order
    private fun steps(): List<Pair<Double, String>> = kv["steps"]?.split(",")?.filter { it.isNotBlank() }?.map { w ->
        val i = w.indexOf(':')
        val t = if (i > 0) w.substring(0, i).toDoubleOrNull() else null
        if (t == null || t < 0) fail("LIZ_SENDER_TEST: step \"$w\" is not <seconds>:<action>")
        t to w.substring(i + 1)
    }?.sortedBy { it.first } ?: emptyList()

    // ai: one step as its control does it; true while the run is stopped by a step (a start ends that)
    private fun step(s: SendState, act: String, stopped: Boolean): Boolean {
        val i = act.indexOf('=')
        val k = if (i > 0) act.substring(0, i) else act
        val v = if (i > 0) act.substring(i + 1) else ""
        fun n(r: IntRange) = v.toIntOrNull()?.takeIf { it in r } ?: fail("LIZ_SENDER_TEST: step $act")
        when (k) {
            "full" -> s.toggleFull()
            "max" -> s.maximize?.invoke()
            "rail" -> s.toggleRail()
            "pause" -> s.pause(true)
            "resume" -> s.pause(false)
            "stop" -> { s.stop(); return true }
            "start" -> { s.start(); return stopped }
            "press" -> { s.requestStart(); return stopped }
            "confirm" -> { s.answerBrightness(true); return stopped }
            "cancel" -> s.answerBrightness(false)
            "size" -> { s.chooseSize(n(25..100)); s.commitSize() }
            "spacing" -> { s.chooseSpacing(n(Pick.SPACING_MIN..Pick.SPACING_MAX)); s.repick() }
            "margin" -> { s.chooseMargin(n(Pick.QUIET..Pick.MARGIN_MAX)); s.repick() }
            "codes" -> s.chooseCodes(n(1..2))
            "blocks" -> { s.chooseBlocks(n(0..128)); s.repick() }
            "fps" -> { s.chooseFps(n(1..60)); s.commitFps() }
            "vsync" -> s.chooseVsync(n(0..1) == 1)
            "enc" -> if (v in setOf("auto", "gpu", "cpu")) s.chooseEnc(v) else fail("LIZ_SENDER_TEST: step $act")
            "add" -> { s.pick(v.split(java.io.File.pathSeparatorChar).filter { it.isNotEmpty() }.map { java.io.File(it) }); return true }
            "remove" -> { s.items.firstOrNull { it.name == v }?.let { s.remove(it.key) } ?: fail("LIZ_SENDER_TEST: no item $v"); return true }
            "clear" -> { s.clearPicks(); return true }
            "list" -> System.err.println("list: ${(s.items.map { "${it.name} (${if (it.dir) "${it.files} files, " else ""}${it.size} B${if (it.links > 0) ", ${it.links} links skipped" else ""})" } + s.stateLine).joinToString(" | ")}")
            else -> fail("LIZ_SENDER_TEST: unknown step $act")
        }
        return stopped
    }
}
