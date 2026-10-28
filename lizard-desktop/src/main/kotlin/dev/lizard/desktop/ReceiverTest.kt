package dev.lizard.desktop

import kotlinx.coroutines.delay
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlin.system.exitProcess

// ai: The receiver's integration hook (2026-10-25, SenderTest.kt's twin): env LIZ_RECEIVER_TEST, space separated
// ai: key=value, e.g. "secs=20 decoder=cpu". Every setting starts at its default (the kept ones neither read nor
// ai: written: Prefs.forget; the run's own held in memory), the ones named applied; the app opens on Receive and starts
// ai: the camera, and after secs (10 unless named) counted from the receiver's first answer (the phase On) one JSON
// ai: line {"rx": <the newest stats>, "received": <the received files' list, Received.kt>, "camera": <the camera open
// ai: as the session describes it (rx_session.h camera: its controls' values), {} none>} goes to stdout and the app
// ai: exits 0; any failure prints "FAIL <why>" and exits 1, the camera stopped and the session destroyed first. Keys:
// ai: secs; camera (an id, rxCameras': it must be listed, but upload:<a .lizard>, the Upload camera, whose file's end
// ai: ends the run as secs does; unless named the simulated camera sim:0 where LIZ_CAMERA_SIM
// ai: is set, else the first listed); mode (an index into the camera's modes, or WxH@fps); layout 1:1 or 2:1; decoder
// ai: auto, gpu or cpu; steps, comma separated <seconds>:<action> from the first answer, each as its control does it:
// ai: start, stop (Start and Stop camera), swap (the bar's swap: to Send, or back to Receive), home (Home), receive
// ai: (Home's Receive), layout=1:1|2:1, decoder=auto|gpu|cpu, res=<W>x<H>, format=<name> and fps=<rate> (Settings' menus), zoom=<v>,
// ai: exposure=<us>|auto and gain=<v>|auto (Settings' bars, a value the camera's own units); logged on stderr as "step
// ai: <s>: <action>". Each second's stats go to stderr as "rx: <stats JSON>". A camera stopped by a step, away from
// ai: Receive, or by itself once a file is in, is no failure; one stopped by an error is.
class ReceiverTest private constructor(private val secs: Double, private val kv: Map<String, String>) {
    companion object {
        private val KEYS = setOf("secs", "camera", "mode", "layout", "decoder", "steps")
        // ai: how long the camera and the receiver may take to answer (a cold pipeline build on a slow device)
        private const val START_SECS = 60.0

        fun parse(env: String?): ReceiverTest? {
            if (env == null) return null
            val kv = env.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.associate { w ->
                val i = w.indexOf('=')
                if (i <= 0) fail("LIZ_RECEIVER_TEST: \"$w\" is not key=value")
                w.substring(0, i) to w.substring(i + 1)
            }
            (kv.keys - KEYS).let { if (it.isNotEmpty()) fail("LIZ_RECEIVER_TEST: unknown ${it.joinToString()}") }
            val secs = kv["secs"]?.let { it.toDoubleOrNull()?.takeIf { s -> s > 0 } ?: fail("LIZ_RECEIVER_TEST: secs=$it") } ?: 10.0
            kv["layout"]?.let { if (it != "1:1" && it != "2:1") fail("LIZ_RECEIVER_TEST: layout=$it") }
            kv["decoder"]?.let { if (it !in setOf("auto", "gpu", "cpu")) fail("LIZ_RECEIVER_TEST: decoder=$it") }
            return ReceiverTest(secs, kv)
        }

        // ai: the session ended first (the camera stopped, its controls put back), so a FAIL is exit 1 and never a
        // ai: teardown under a stream; the close is the UI thread's, asked there
        var running: App? = null
        fun fail(why: String): Nothing {
            println("FAIL $why")
            System.out.flush()
            running?.let { a ->
                runCatching { if (java.awt.EventQueue.isDispatchThread()) a.close() else java.awt.EventQueue.invokeAndWait { a.close() } }
            }
            exitProcess(1)
        }
    }

    // ai: the settings named, before the window; no camera named, the simulated one where LIZ_CAMERA_SIM sets it up, so
    // ai: a run never streams a real camera by default (2026-10-25, as V4L2 came to probe every /dev/video*)
    fun apply(app: App) {
        val r = app.recv
        kv["layout"]?.let { r.layout = it }
        kv["decoder"]?.let { r.decoder = it }
        (kv["camera"] ?: if (System.getenv("LIZ_CAMERA_SIM") != null) "sim:0" else null)?.let { r.testCamera = it }
        kv["mode"]?.let { r.testMode = it }
    }

    // ai: on the UI thread (App.kt's LaunchedEffect)
    suspend fun run(app: App) {
        running = app
        val r = app.recv
        if (app.loadError.isNotEmpty()) fail(app.loadError)
        r.enter()
        val t0 = System.nanoTime()
        while (r.phase != ReceiveState.Phase.On) {
            (r.phase as? ReceiveState.Phase.Error)?.let { fail(it.why) }
            // ai: a file whole in the receiver's first answer stops the camera in the poll that saw it (On, then Idle
            // ai: in one tick: a 255 KB bundle on the simulated camera, 2026-10-25), which is no failure
            if (r.phase == ReceiveState.Phase.Idle) { if (r.rx.verified) break; fail("the camera did not start") }
            if (System.nanoTime() - t0 > START_SECS * 1e9) fail("no answer from the receiver after ${START_SECS.toInt()} s (${r.phase}, ${r.raw.ifEmpty { "no stats" }})")
            delay(50)
        }
        System.err.println("rx: camera ${r.camera}, mode ${r.mode?.let { "${it.w}x${it.h} ${it.format} at ${ReceiveState.fpsText(it.fps)}" } ?: "?"}")
        val begun = System.nanoTime()
        val end = begun + (secs * 1e9).toLong()
        val todo = steps().toMutableList()
        var away = false        // ai: the camera stopped by a step, or Receive left
        var nextLine = begun
        while (System.nanoTime() < end) {
            (r.phase as? ReceiveState.Phase.Error)?.let { fail(it.why) }
            val at = (System.nanoTime() - begun) / 1e9
            while (todo.isNotEmpty() && todo.first().first <= at) {
                val (t, act) = todo.removeAt(0)
                System.err.println("step %.1f: %s".format(java.util.Locale.ROOT, t, act))
                away = step(app, act, away)
            }
            if (away && r.running && app.screen == Screen.Receive) away = false
            if (!r.running && !away && !r.rx.verified) { if (r.upload) break else fail("the camera stopped") }
            val now = System.nanoTime()
            if (now >= nextLine) { nextLine = now + 1_000_000_000L; if (r.raw.isNotEmpty()) System.err.println("rx: ${r.raw}") }
            delay(100)
        }
        (r.phase as? ReceiveState.Phase.Error)?.let { fail(it.why) }
        val stats: JsonElement = dev.lizard.desktop.parse(r.raw.ifEmpty { "{}" }) ?: JsonObject(emptyMap())   // ai: the JSON helper, not this class's parse
        val cam: JsonElement = dev.lizard.desktop.parse(r.cameraNow()) ?: JsonObject(emptyMap())
        // ai: the list read once the close has waited for the filings (a tar's unpack may still run when the time is up)
        app.close()
        val line = JsonObject(mapOf("rx" to stats, "received" to JsonArray(Received.list().map { Received.asJson(it) }),
            "camera" to cam)).toString()
        println(line)
        System.out.flush()
        exitProcess(0)
    }

    // ai: the steps as (seconds, action), in order
    private fun steps(): List<Pair<Double, String>> = kv["steps"]?.split(",")?.filter { it.isNotBlank() }?.map { w ->
        val i = w.indexOf(':')
        val t = if (i > 0) w.substring(0, i).toDoubleOrNull() else null
        if (t == null || t < 0) fail("LIZ_RECEIVER_TEST: step \"$w\" is not <seconds>:<action>")
        t to w.substring(i + 1)
    }?.sortedBy { it.first } ?: emptyList()

    // ai: one step as its control does it; true while the camera is stopped by a step or Receive is left
    private fun step(app: App, act: String, away: Boolean): Boolean {
        val r = app.recv
        val i = act.indexOf('=')
        val k = if (i > 0) act.substring(0, i) else act
        val v = if (i > 0) act.substring(i + 1) else ""
        when (k) {
            "start" -> { r.start(); return away }
            "stop" -> { r.stop(); return true }
            "swap" -> { app.go(if (app.screen == Screen.Send) Screen.Receive else Screen.Send); return app.screen != Screen.Receive || away }
            "home" -> { app.go(Screen.Home); return true }
            "receive" -> { app.go(Screen.Receive); return away }
            "layout" -> if (v == "1:1" || v == "2:1") r.chooseLayout(v) else fail("LIZ_RECEIVER_TEST: step $act")
            "decoder" -> if (v in setOf("auto", "gpu", "cpu")) r.chooseDecoder(v) else fail("LIZ_RECEIVER_TEST: step $act")
            // ai: Resolution and Frame rate as their menus take them: a size or a rate the camera offers (a restart; the
            // ai: mode key's choice gives way to the menus' from then on)
            "res" -> if (r.info?.modes?.any { it.size == v } == true) { r.testMode = null; r.chooseRes(v) } else fail("LIZ_RECEIVER_TEST: step $act: no such size")
            "format" -> if (r.info?.modes?.any { it.size == r.mode?.size && it.format == v } == true) { r.testMode = null; r.chooseFormat(v) }
                else fail("LIZ_RECEIVER_TEST: step $act: no such format at this size")
            "fps" -> v.toDoubleOrNull()?.takeIf { f -> r.info?.modes?.any { it.size == r.mode?.size && it.format == r.mode?.format && kotlin.math.abs(it.fps - f) < 0.01 } == true }
                ?.let { r.testMode = null; r.chooseFps(it) } ?: fail("LIZ_RECEIVER_TEST: step $act: no such rate at this size")
            // ai: the bars as a drag leaves them: a value in the camera's units, or auto (the first step) for exposure and gain
            "zoom" -> v.toDoubleOrNull()?.let { r.zoomTo(it) } ?: fail("LIZ_RECEIVER_TEST: step $act")
            "exposure" -> if (v == "auto") r.exposureTo(null) else v.toDoubleOrNull()?.let { r.exposureTo(it) } ?: fail("LIZ_RECEIVER_TEST: step $act")
            "gain" -> if (v == "auto") r.gainTo(null) else v.toDoubleOrNull()?.let { r.gainTo(it) } ?: fail("LIZ_RECEIVER_TEST: step $act")
            else -> fail("LIZ_RECEIVER_TEST: unknown step $act")
        }
        return away
    }
}
