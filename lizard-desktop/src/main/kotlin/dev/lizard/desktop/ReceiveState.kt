package dev.lizard.desktop

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asComposeImageBitmap
import dev.lizard.readout.Readout
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import org.jetbrains.skia.Bitmap
import org.jetbrains.skia.ColorAlphaType
import org.jetbrains.skia.ColorType
import org.jetbrains.skia.ImageInfo
import java.awt.EventQueue
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import javax.swing.Timer
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.roundToInt

// ai: The receiver (2026-10-25, the PC receiver; the Android app's Engine and MainActivity, the web's recv.mjs): one
// ai: session (Native.rx*, native/rx_session.h) made on the first start, a camera in it and its stream into the
// ai: receiver, what Receive shows. Every session call that can wait (create, configure, open, start, stop, set,
// ai: forget, the file taken out of the store) runs on one thread of its own ("lizard-rx") in the order asked, so a
// ai: start, a stop and a receiver's rebuild never interleave; the stats a quarter second and the preview about 30
// ai: times a second on the UI thread (rxStats, rxPreview, which wait on nothing). The phases are the app's
// ai: Engine.Phase: Starting (the session made, the camera opened and its stream begun), Loading (the stream runs, the
// ai: receiver not yet past "starting": its first plan, or a new receiver for another decoder or crop), On, Idle,
// ai: Error. A result of work begun before a newer start or stop is dropped (gen). The session and any transfer partly
// ai: in outlive a stop and the screen: leaving Receive stops the camera and closes it (another app may want it); the
// ai: app's close destroys the session, the camera's controls put back by its stop. Settings: the camera (stop, open,
// ai: start), resolution and rate (stop, start), decoder and crop (a new receiver, rxConfigure: the camera stopped and
// ai: started again around it in its mode), zoom, exposure and gain (rxSet whenever a camera is open: the session sets
// ai: it at once while the camera runs, keeps it while it does not, and sets every one kept again after each start,
// ai: since each backend puts back at stop what was set; the kept ones handed to it at each open), batch size
// ai: (rxBatchCap); each kept (Prefs.Rx). A verified file is taken out of the store and filed into Downloads
// ai: (Received.kt) where no list entry holds its root and no filing of it is under way (what is filed decides, as the
// ai: app's keep: a file deleted, or moved out of Downloads, and received again is filed again), and the camera stops
// ai: then (once a root, as the web's and the app's, so a start after it reads on). The filings, Home's deletes and a
// ai: replay's end run on a thread of their own ("lizard-files") that the app's close waits for. A camera whose stream
// ai: ended by itself, or that did not start, is closed as well as stopped, so the next Start opens it again by its id
// ai: (one unplugged and plugged in again is a new device behind the same id). The figures of the receiver's last
// ai: second show only while they are this stream's (the stats' "live", rx_session.h): none while the camera is off, as
// ai: the web and the Android app show none, a received file's line kept.
class ReceiveState(private val app: App) {
    sealed interface Phase {
        data object Idle : Phase
        data object Starting : Phase
        data object Loading : Phase
        data object On : Phase
        data class Error(val why: String) : Phase
    }

    // ai: what the session says of the cameras (rx_session.h camera(), rxCameras): a camera listed, a mode, a control
    class Cam(val id: String, val name: String, val backend: String, val detail: String, val label: String)
    class Mode(val index: Int, val w: Int, val h: Int, val fps: Double, val fpsMin: Double, val format: String, val compressed: Boolean) {
        val size get() = "${w}x$h"
    }
    class Control(val id: String, val name: String, val unit: String, val min: Double, val max: Double, val step: Double, val def: Double,
                  val value: Double, val canAuto: Boolean, val isAuto: Boolean)
    class CamInfo(val id: String, val name: String, val backend: String, val detail: String, val modes: List<Mode>, val pick: Int,
                  val controls: List<Control>) {
        fun control(id: String) = controls.firstOrNull { it.id == id && it.max > it.min }
    }

    // ai: Save replays (the app's): a run's folder of the newest frames the decoder takes, Native.data/replays/<run>
    enum class RunState { Recording, Ending, Ready, Failed }
    data class Run(val run: String, val dir: File, val handle: Long, val state: RunState, val frames: Long = 0, val bytes: Long = 0, val why: String = "")

    var phase by mutableStateOf<Phase>(Phase.Idle)
    var cameras by mutableStateOf<List<Cam>>(emptyList())
    var camera by mutableStateOf(Prefs.Rx.camera)              // ai: the camera chosen ("": the first listed)
    var info by mutableStateOf<CamInfo?>(null)                 // ai: the camera open
    var mode by mutableStateOf<Mode?>(null)                    // ai: its mode, running or next
    var decoder by mutableStateOf(Prefs.Rx.decoder)
    var layout by mutableStateOf(Prefs.Rx.layout)
    var batch by mutableStateOf(Prefs.Rx.batch)
    var zoom by mutableStateOf<Double?>(null)                  // ai: the camera's settings as Settings shows them (null: its own)
    var exposure by mutableStateOf<Double?>(null)
    var gain by mutableStateOf<Double?>(null)
    var raw by mutableStateOf("")                              // ai: the newest stats JSON as the session wrote it
    var stats by mutableStateOf<JsonObject?>(null)
    var rx by mutableStateOf(Readout.Rx())
    var root by mutableStateOf("")                             // ai: the transfer's root the receiver holds
    var autoRan by mutableStateOf("")                          // ai: "GPU" or "CPU": what auto runs on
    var preview by mutableStateOf<ImageBitmap?>(null)
    var settingsOpen by mutableStateOf(Prefs.Rx.settingsOpen)
    var toolsOpen by mutableStateOf(Prefs.Rx.toolsOpen)
    var collapsed by mutableStateOf(Prefs.Rx.collapsed)
    var showStats by mutableStateOf(Prefs.Rx.stats)
    // ai: Save replays: never kept, as the Android app's (Settings.replays): a recording left on costs every later run its lag
    var replays by mutableStateOf(false)
    var files by mutableStateOf<List<Received.Entry>>(emptyList())
    // ai: why the newest file is not in Downloads (Received.Filed's why; it is in the app's own folder then, listed from
    // ai: there), "" where it is; shown under the state line until a later filing of it puts it there
    var fileNote by mutableStateOf("")
    var partials by mutableStateOf<List<Partials.Entry>>(emptyList())
    val runs = mutableStateListOf<Run>()

    // ai: the test hook's (ReceiverTest.kt): a camera that must be listed, and a mode by index or "WxH@fps"
    var testCamera: String? = null
    var testMode: String? = null

    val running get() = phase == Phase.Starting || phase == Phase.Loading || phase == Phase.On
    val on get() = phase == Phase.On
    // ai: the receiver's figures of its last second are this stream's (rx_session.h stats "live"): false while the
    // ai: camera is off; from a start the receiver's own, 0 until the stream's first second has closed
    val live get() = on && stats?.flag("live") == true
    val testing get() = on && rx.state == "test"
    // ai: the received file the receiver holds, once it is in the list; none while the test stream is read
    val kept: Received.Entry? get() = if (rx.verified && !testing && root.isNotEmpty()) files.firstOrNull { it.root == root } else null
    val cpu get() = decoder == "cpu" || (decoder == "auto" && autoRan == "CPU")

    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "lizard-rx").apply { isDaemon = true } }
    // ai: The received files' own work, one at a time in the order asked: a filing (Received.file: the file into
    // ai: Downloads, a tar unpacked), the takes filed again (Received.retry), Home's deletes, an unfinished transfer's
    // ai: folder deleted with no session, a replay's end. close() waits for it: the app's exit, once the window is gone,
    // ai: cut each short before (a half-unpacked bundle in Downloads, an entry delisted with its file left).
    private val filer = Executors.newSingleThreadExecutor { r -> Thread(r, "lizard-files").apply { isDaemon = true } }
    @Volatile private var h = 0L
    @Volatile private var closed = false
    private var gen = 0                       // ai: UI thread: a newer start or stop orphans what was begun before it
    private var sessionMade = false           // ai: UI thread: the session exists
    private var configuring = false           // ai: UI thread: a new receiver being made, its stats not yet the new one's
    // ai: the session's thread alone
    private var opened = ""                   // ai: the camera open in the session ("" none: open never succeeded, or closed)
    private var openInfo: CamInfo? = null
    private var made = ""                     // ai: "<decoder>/<layout>" the receiver was made for ("" not known)
    private var streaming = false

    private class Refused(why: String) : Exception(why)
    private class Want(val camera: String?, val strict: Boolean, val mode: String?, val decoder: String, val layout: String, val batch: Int)

    private fun post(my: Int = -1, f: () -> Unit) = EventQueue.invokeLater { if (my < 0 || my == gen) f() }
    private fun task(f: () -> Unit) = worker.execute {
        try { f() } catch (e: Throwable) { System.err.println("receive: ${e.javaClass.simpleName}: ${e.message}") }
    }
    // ai: on the files' thread; once the close has let it go, said and not done (a take not filed is filed at the next
    // ai: launch, Received.retry)
    private fun later(f: () -> Unit) {
        try {
            filer.execute { try { f() } catch (e: Throwable) { System.err.println("receive: ${e.javaClass.simpleName}: ${e.message}") } }
        } catch (_: RejectedExecutionException) { System.err.println("receive: the app is closing: left for the next launch") }
    }

    // ai: the session (the session's thread): made at the first start, a new receiver where the decoder or the crop
    // ai: changed (the finished file taken out of the store first: a new receiver lets its finished folders go)
    private fun session(dec: String, lay: String): Long {
        if (closed) throw Refused("the app is closing")
        val key = "$dec/$lay"
        val hh = h
        if (hh != 0L) {
            if (made != key) {
                takeOut()
                val why = Native.rxConfigure(hh, dec, lay)
                // ai: a refusal leaves the receiver in hand unknown here (rx_session.h reconfigure: the new one where the
                // ai: camera did not start again, the old one where the new one could not be made), so the next call
                // ai: configures again, which the session ends at once where the receiver in hand already matches
                if (why.isNotEmpty()) { made = ""; throw Refused(why) }
                made = key
            }
            return hh
        }
        val store = File(Native.data, "store").apply { mkdirs() }
        val cache = File(Native.buildDir, "pipeline").apply { mkdirs() }
        val made1 = Native.rxCreate(Native.assets, cache.path, store.path, dec, lay)
        if (made1 == 0L) throw Refused(Native.txError().ifEmpty { "no receiver" })
        h = made1
        made = key
        post { sessionMade = true; timer.start() }
        return made1
    }

    // ai: the cameras every backend sees, labelled for the menu: the device's name, a number after a name two share
    private fun listed(): List<Cam> {
        val j = parse(Native.rxCameras()) ?: return emptyList()
        val notes = (j["notes"] as? JsonArray)?.mapNotNull { (it as? JsonPrimitive)?.contentOrNull }.orEmpty().joinToString("; ")
        if (notes != lastNotes) { lastNotes = notes; if (notes.isNotEmpty()) System.err.println("receive: cameras: $notes") }
        val cams = (j["cameras"] as? JsonArray)?.mapNotNull { it as? JsonObject }.orEmpty().map { o ->
            Cam(o.str("id"), o.str("name"), o.str("backend"), o.str("detail"), o.str("name").ifEmpty { o.str("id") })
        }
        val seen = HashMap<String, Int>()
        return cams.map { c ->
            val k = (seen[c.label] ?: 0) + 1
            seen[c.label] = k
            if (k == 1) c else Cam(c.id, c.name, c.backend, c.detail, "${c.label} $k")
        }
    }
    @Volatile private var lastNotes = ""
    // ai: a phone's crop, from its listing's detail ("2:1, <ip>"; camera/phone.cpp), null for any other camera
    private fun phoneLayout(cams: List<Cam>, id: String) =
        cams.firstOrNull { it.id == id && it.backend == "phone" }?.detail?.substringBefore(",")?.trim()?.takeIf { it == "1:1" || it == "2:1" }
    val phone get() = info?.backend == "phone" || camera.startsWith("phone:")
    // ai: A phone by its address (2026-10-27, beside the beacon and the gateways, which a firewall or a route may defeat): the
    // ai: Camera menu's last entry asks for it (askAddress); the address typed is the camera's id, phone:<ip>:<port>
    // ai: (the stream's port where none is typed), listed by it while no listing has it
    var askAddress by mutableStateOf(false)
    val typedPhone: String? get() = camera.takeIf { c -> c.startsWith("phone:") && cameras.none { it.id == c } }
    val cameraOptions: List<Pair<String, String>> get() =
        cameras.map { it.id to it.label } + listOfNotNull(typedPhone?.let { it to "Phone at ${it.removePrefix("phone:").removeSuffix(":$PHONE_PORT")}" }) +
            ("phone:" to "Phone by address")
    fun chooseCameraOption(id: String) { if (id == "phone:") askAddress = true else { askAddress = false; chooseCamera(id) } }
    fun choosePhoneAddress(typed: String) {
        val a = typed.trim()
        if (!Regex("\\d{1,3}(\\.\\d{1,3}){3}(:\\d{1,5})?").matches(a)) return
        askAddress = false
        chooseCamera("phone:" + if (':' in a) a else "$a:$PHONE_PORT")
    }

    private fun camInfo(s: String): CamInfo? {
        val j = parse(s) ?: return null
        val i = j["info"] as? JsonObject ?: return null
        val modes = (j["modes"] as? JsonArray).orEmpty().mapIndexedNotNull { k, e ->
            (e as? JsonObject)?.let { Mode(k, it.num("w").toInt(), it.num("h").toInt(), it.num("fps"), it.num("fpsMin"), it.str("format"), it.flag("compressed")) }
        }
        val controls = (j["controls"] as? JsonArray).orEmpty().mapNotNull { e ->
            (e as? JsonObject)?.let { c -> Control(c.str("id"), c.str("name"), c.str("unit"), c.num("min"), c.num("max"), c.num("step"), c.num("def"),
                c.num("value"), c.flag("canAuto"), c.flag("isAuto")) }
        }
        return CamInfo(i.str("id"), i.str("name"), i.str("backend"), i.str("detail"), modes, (j["pick"] as? JsonPrimitive)?.contentOrNull?.toDoubleOrNull()?.toInt() ?: -1,
            controls)
    }

    // ai: the camera open (the one before stopped and closed by the session), what it offers read, and what is kept for
    // ai: it handed to the session (its zoom for the crop, its exposure and gain), which sets each after every start.
    // ai: The session closes the camera in hand before it opens another, whether or not that open succeeds, so none is
    // ai: named open here until both the open and its description have come back.
    private fun openCam(hh: Long, id: String, lay: String): CamInfo {
        openInfo?.takeIf { opened == id }?.let { return it }
        if (streaming) { Native.rxStop(hh); streaming = false }
        opened = ""; openInfo = null
        val why = Native.rxOpen(hh, id)
        if (why.isNotEmpty()) throw Refused(why)
        val ci = camInfo(Native.rxCamera(hh)) ?: run { Native.rxClose(hh); throw Refused("the camera says nothing of itself") }
        opened = id
        openInfo = ci
        ci.control("zoom")?.let { c -> Prefs.Rx.zoom(id, lay)?.let { setNow(hh, c.id, it, false) } }
        ci.control("exposure")?.let { c -> Prefs.Rx.exposure(id)?.let { setNow(hh, c.id, it, false) } }
        ci.control("gain")?.let { c -> Prefs.Rx.gain(id)?.let { setNow(hh, c.id, it, false) } }
        return ci
    }
    private fun closeCam() {
        val hh = h
        if (hh != 0L && opened.isNotEmpty()) { halt(); Native.rxClose(hh) }
        opened = ""; openInfo = null
    }

    // ai: the mode kept for a camera (its resolution, rate and format where it offers them; a format not kept,
    // ai: uncompressed first), else the session's pick (camera.h pick: 1920 x 1080 at 60 or the nearest)
    private fun keptMode(ci: CamInfo): Mode {
        val res = Prefs.Rx.res(ci.id)
        val fps = Prefs.Rx.fps(ci.id)
        val at = ci.modes.filter { it.size == res && abs(it.fps - fps) < 0.01 }
        return at.firstOrNull { it.format == Prefs.Rx.format(ci.id) } ?: at.minByOrNull { if (it.compressed) 1 else 0 }
            ?: ci.modes.getOrNull(ci.pick) ?: ci.modes.firstOrNull() ?: throw Refused("the camera offers no mode")
    }
    // ai: the test hook's: an index, or "WxH@fps"
    private fun modeOf(ci: CamInfo, spec: String): Mode? {
        spec.toIntOrNull()?.let { return ci.modes.getOrNull(it) }
        val m = Regex("(\\d+)x(\\d+)@([0-9.]+)").matchEntire(spec) ?: return null
        val (w, hh, f) = m.destructured
        return ci.modes.filter { it.w == w.toInt() && it.h == hh.toInt() && abs(it.fps - f.toDouble()) < 0.01 }.minByOrNull { if (it.compressed) 1 else 0 }
    }

    // ai: Start camera (the session's thread): the cameras listed, the session, the chosen one opened (the kept id where
    // ai: listed, else the first) with its kept settings handed over, its mode started (the session sets them then). A
    // ai: start that fails closes the camera too: a device unplugged leaves
    // ai: the camera open on what the system tore down (V4L2's fd answers ENODEV for good, AVFoundation's device object
    // ai: never connects again), and the next Start opens it again by its id.
    private fun begin(my: Int, w: Want) {
        try {
            // ai: the cameras first: no receiver is made (the GPU's pipelines) where none is listed
            val cams = listed()
            post { cameras = cams }
            // ai: a phone by its address opens unlisted
            val typed = w.camera?.takeIf { it.startsWith("phone:") }
            if (cams.isEmpty() && typed == null) { post(my) { phase = Phase.Error("No camera was found.") }; return }
            val id = w.camera?.takeIf { c -> c == typed || cams.any { it.id == c } } ?: if (w.strict) throw Refused("${w.camera} is not listed") else cams[0].id
            // ai: a phone sends its own crop (liblizard/core/rx/stream.h): the receiver crops as the phone did, from its
            // ai: listing, else (an address typed) from its hello once open, the receiver made again for it
            var lay = phoneLayout(cams, id) ?: w.layout
            var hh = session(w.decoder, lay)
            Native.rxBatchCap(hh, w.batch)
            val ci = openCam(hh, id, lay)
            if (ci.backend == "phone") ci.detail.substringBefore(",").trim().takeIf { (it == "1:1" || it == "2:1") && it != lay }?.let { l ->
                lay = l
                hh = session(w.decoder, l)
            }
            if (lay != w.layout) post { layout = lay }
            val m = w.mode?.let { modeOf(ci, it) ?: throw Refused("the camera offers no mode $it") } ?: keptMode(ci)
            if (streaming) { Native.rxStop(hh); streaming = false }
            val why = Native.rxStart(hh, m.index)
            if (why.isNotEmpty()) throw Refused(why)
            streaming = true
            post { info = ci }
            post(my) {
                camera = id; mode = m
                zoom = Prefs.Rx.zoom(id, lay) ?: ci.control("zoom")?.value
                exposure = Prefs.Rx.exposure(id); gain = Prefs.Rx.gain(id)
                phase = Phase.Loading
            }
        } catch (e: Throwable) {
            runCatching { closeCam() }
            post(my) { phase = Phase.Error("The camera did not start: ${e.message ?: e.javaClass.simpleName}") }
        }
    }
    private fun setNow(hh: Long, id: String, v: Double, auto: Boolean) {
        val why = Native.rxSet(hh, id, v, auto)
        if (why.isNotEmpty()) System.err.println("receive: $id: $why")
    }
    // ai: the camera open as the session describes it now (rxCamera: its controls' values as last read; never waits),
    // ai: "{}" with none (the test hook's last line)
    fun cameraNow(): String = h.takeIf { it != 0L }?.let { runCatching { Native.rxCamera(it) }.getOrNull() } ?: "{}"
    // ai: the stream ended (the session's thread): the camera's stop
    private fun halt() {
        val hh = h
        if (hh != 0L && streaming) Native.rxStop(hh)
        streaming = false
    }

    fun start() {
        if (app.loadError.isNotEmpty()) { phase = Phase.Error(app.loadError); return }
        if (running) return
        val my = ++gen
        phase = Phase.Starting
        configuring = false
        // ai: a Start tries again what did not go: a take that failed, and the files Downloads did not take
        skip = ""
        retryKept()
        val w = Want(testCamera ?: camera.ifEmpty { null }, testCamera != null, testMode, decoder, layout, batch)
        task { begin(my, w) }
    }
    fun stop() {
        ++gen
        previewTimer.stop()
        if (running) phase = Phase.Idle
        task { halt() }
    }
    fun toggle() = if (running) stop() else start()
    // ai: Receive entered (App.go): the lists read again, the camera started
    fun enter() { refreshLists(); start() }
    // ai: Receive left: the camera stopped and closed; the session and its transfer kept
    fun leave() { stop(); task { closeCam() } }

    // ai: the app's close: everything stopped and the session destroyed (the camera's controls put back by its stop), a
    // ai: recording run ended first and a file verified since the last poll taken out (the next launch's store would let
    // ai: it go); waited for a few seconds at most. Then the files' thread: the filing under way and any just asked, the
    // ai: deletes, a replay's end, CLOSE_FILING_SECS at most (the window is hidden by then, App.close); a filing still
    // ai: under way after that is done again from the app's folder at the next launch (Received.retry), its half-made
    // ai: unpack never in place.
    fun close() {
        timer.stop(); previewTimer.stop(); ++gen
        closed = true
        val rec = runs.lastOrNull { it.state == RunState.Recording }
        val last = raw; val rowsNow = rows.joinToString("\n"); val more = moreJson()
        val done = CountDownLatch(1)
        worker.execute {
            try {
                val hh = h
                if (rec != null && hh != 0L) { Native.rxRecord(hh, 0); runCatching { Native.rxReplayEnd(rec.handle, last, rowsNow, more) } }
                runCatching { takeOut() }
                closeCam()
                if (hh != 0L) { Native.rxDestroy(hh); h = 0L }
            } catch (_: Throwable) {
            } finally { done.countDown() }
        }
        done.await(8, TimeUnit.SECONDS)
        filer.shutdown()
        if (!filer.awaitTermination(CLOSE_FILING_SECS, TimeUnit.SECONDS)) System.err.println("receive: a filing still under way at the close: done at the next launch")
    }

    // ai: Settings
    fun chooseCamera(id: String) {
        if (id == camera && info?.id == id) return
        camera = id; Prefs.Rx.camera = id
        layout = phoneLayout(cameras, id) ?: Prefs.Rx.layout   // ai: a phone's crop its own; another camera's the one kept
        zoom = null; exposure = Prefs.Rx.exposure(id); gain = Prefs.Rx.gain(id)
        if (running) { stop(); start() }
        else if (sessionMade) {
            val lay = layout
            val my = gen
            task {
                val hh = h
                if (hh == 0L) return@task
                runCatching { openCam(hh, id, lay) }.onSuccess { ci ->
                    val m = runCatching { keptMode(ci) }.getOrNull()
                    post { if (camera == id) { info = ci; mode = m; zoom = Prefs.Rx.zoom(id, layout) ?: ci.control("zoom")?.value } }
                    // ai: a camera chosen and open: an error before it said is no longer the line's
                    post(my) { if (camera == id && phase is Phase.Error) phase = Phase.Idle }
                }.onFailure { e ->
                    System.err.println("receive: $id: ${e.message}")
                    // ai: the session closed the camera in hand before this open: none is shown as open
                    post(my) { if (camera == id) { info = null; mode = null; phase = Phase.Error("The camera did not start: ${e.message ?: e.javaClass.simpleName}") } }
                }
            }
        }
    }
    // ai: Resolution, Format and Frame rate (Settings' menus): each keeps what it can of the other two, the format first
    // ai: (a size or rate the format has not falls back to the rate kept, uncompressed first, then the fastest)
    fun chooseRes(res: String) {
        val ci = info ?: return
        val at = ci.modes.filter { it.size == res }
        val want = mode?.fps ?: 0.0
        val fmt = mode?.format
        val m = at.firstOrNull { it.format == fmt && abs(it.fps - want) < 0.01 }
            ?: at.filter { it.format == fmt }.maxByOrNull { it.fps }
            ?: at.filter { abs(it.fps - want) < 0.01 }.minByOrNull { if (it.compressed) 1 else 0 }
            ?: at.sortedWith(compareByDescending<Mode> { it.fps }.thenBy { it.compressed }).firstOrNull() ?: return
        pickMode(ci, m)
    }
    fun chooseFormat(format: String) {
        val ci = info ?: return
        val cur = mode ?: return
        val at = ci.modes.filter { it.size == cur.size && it.format == format }
        val m = at.firstOrNull { abs(it.fps - cur.fps) < 0.01 } ?: at.maxByOrNull { it.fps } ?: return
        pickMode(ci, m)
    }
    fun chooseFps(f: Double) {
        val ci = info ?: return
        val cur = mode ?: return
        val at = ci.modes.filter { it.size == cur.size && abs(it.fps - f) < 0.01 }
        val m = at.firstOrNull { it.format == cur.format } ?: at.minByOrNull { if (it.compressed) 1 else 0 } ?: return
        pickMode(ci, m)
    }
    private fun pickMode(ci: CamInfo, m: Mode) {
        Prefs.Rx.mode(ci.id, m.size, m.fps, m.format)
        mode = m
        if (running) { stop(); start() }
    }
    fun chooseDecoder(d: String) {
        if (d == decoder) return
        decoder = d; Prefs.Rx.decoder = d
        if (d != "auto") autoRan = ""
        reconfigure()
    }
    fun chooseLayout(l: String) {
        if (l == layout) return
        layout = l; Prefs.Rx.layout = l
        val ci = info
        if (ci != null) zoom = Prefs.Rx.zoom(ci.id, l) ?: zoom
        reconfigure()
        // ai: the crop's own zoom, once its receiver runs (a crop keeps a zoom of its own, the app's 2026-10-18)
        if (ci != null) zoom?.let { z -> ci.control("zoom")?.let { send(it.id, z, false) } }
    }
    // ai: a new receiver for the decoder and crop chosen: on the session's thread, the camera stopped and started again
    // ai: around it by the session; until its stats are its own the line says "Getting ready"
    private fun reconfigure() {
        if (!sessionMade && !running) return
        val dec = decoder; val lay = layout
        val my = gen
        if (running) { configuring = true; phase = Phase.Loading; previewTimer.stop() }
        task {
            try {
                session(dec, lay)
                post { configuring = false }
            } catch (e: Throwable) {
                // ai: a camera that did not start again is closed too, as a start's that fails (begin)
                post { configuring = false; if (gen == my && running) { stop(); task { closeCam() }; phase = Phase.Error("The camera did not start: ${e.message ?: e.javaClass.simpleName}") } }
            }
        }
    }
    fun zoomTo(v: Double) { zoom = v; info?.let { Prefs.Rx.zoom(it.id, layout, v) }; send("zoom", v, false) }
    // ai: null: the camera's own exposure (automatic where it can be)
    fun exposureTo(us: Double?) { exposure = us; info?.let { Prefs.Rx.exposure(it.id, us) }; send("exposure", us ?: 0.0, us == null) }
    fun gainTo(v: Double?) { gain = v; info?.let { Prefs.Rx.gain(it.id, v) }; send("gain", v ?: 0.0, v == null) }
    // ai: a setting moved to the session while a camera is open in it: a drag's values coalesced, the newest set on
    // ai: the session's thread (at once on a running camera; kept by the session otherwise, and set again after each
    // ai: start either way)
    private val pending = ConcurrentHashMap<String, Pair<Double, Boolean>>()
    private fun send(id: String, v: Double, auto: Boolean) {
        if (pending.put(id, v to auto) == null) task {
            val (x, a) = pending.remove(id) ?: return@task
            val hh = h
            if (hh != 0L && opened.isNotEmpty()) setNow(hh, id, x, a)
        }
    }
    fun chooseBatch(n: Int) { batch = n.coerceIn(1, 32); Prefs.Rx.batch = batch }
    fun commitBatch() { val n = batch; task { val hh = h; if (hh != 0L) Native.rxBatchCap(hh, n) } }
    fun toggleSettings() { settingsOpen = !settingsOpen; Prefs.Rx.settingsOpen = settingsOpen }
    fun toggleTools() { toolsOpen = !toolsOpen; Prefs.Rx.toolsOpen = toolsOpen }
    fun toggleRail() { collapsed = !collapsed; Prefs.Rx.collapsed = collapsed }
    fun chooseStats(on: Boolean) { showStats = on; Prefs.Rx.stats = on }
    fun chooseReplays(on: Boolean) { replays = on; replayRefused = false }

    // ai: the cameras read again (the Camera menu opening), off the UI thread: a backend's listing opens each device
    // ai: (V4L2's nodes, Media Foundation's sources)
    private var listing = false
    fun refreshCameras() {
        if (listing || app.loadError.isNotEmpty()) return
        listing = true
        thread(name = "lizard-cameras", isDaemon = true) {
            val l = runCatching { listed() }.getOrNull()
            post { listing = false; if (l != null) cameras = l }
        }
    }

    // ai: Home's lists: the received files (a small file, read here) and the unfinished transfers (a folder each); the
    // ai: takes not in Downloads filed again (retryKept)
    fun refreshLists() {
        files = Received.list()
        retryKept()
        thread(name = "lizard-partials", isDaemon = true) { val p = Partials.list(); post { partials = p } }
    }
    // ai: an unfinished transfer deleted (Home's Delete): forgotten by the receiver where a session runs (the one in
    // ai: hand reset, or its folder removed), else its folder deleted on the files' thread
    fun deletePartial(p: Partials.Entry) {
        partials = partials.filter { it.id != p.id }
        if (sessionMade) task {
            val hh = h
            if (hh != 0L) Native.rxForget(hh, p.root.ifEmpty { p.id }) else p.dir.deleteRecursively()
            val l = Partials.list(); post { partials = l }
        } else later { p.dir.deleteRecursively(); val l = Partials.list(); post { partials = l } }
    }
    // ai: a received file deleted (Home's Delete, Delete all): to the trash, on the files' thread; the receiver forgets it
    // ai: where it holds that root (its transfer and held word, rxClear), so the line's "Received" goes and the same file
    // ai: in the light is received and filed anew (the app's forgetReceived); a deleted root the camera last stopped for
    // ai: stops it again when received again
    fun deleteReceived(e: Received.Entry) = deleteAll(listOf(e))
    fun deleteAll(es: List<Received.Entry> = files) {
        val gone = es.map { it.path }.toSet()
        files = files.filter { it.path !in gone }
        val roots = es.map { it.root }.filter { it.isNotEmpty() }.toSet()
        if (root.isNotEmpty() && root in roots) forget(root)
        if (endedRoot in roots && endedRoot != root) endedRoot = ""
        if (noteRoot in roots) { noteRoot = ""; fileNote = "" }
        later { es.forEach { runCatching { Received.delete(it) } }; val l = Received.list(); post { files = l } }
    }
    // ai: The receiver forgets the root it holds (rxClear, on the session's thread), and only then the screen its stop
    // ai: once a root: a poll between the delete and the clear reads the root still verified, so a reset made before the
    // ai: clear was undone by that poll (the same file received again then never stopped the camera; the app's clearAsked,
    // ai: 2026-10-10)
    private fun forget(r: String) {
        task {
            val hh = h
            if (hh != 0L) Native.rxClear(hh)
            post { if (endedRoot == r) endedRoot = ""; if (skip == r) skip = "" }
        }
    }

    // ai: The verified file out of the store (the session's thread, so no reconfigure or clear comes between), from the
    // ai: stats the session reads now, unless it is in the list already or taken already (no file left in the store):
    // ai: moved into the app's own folder at once (Received.take: a rename beside the store, which no store's open lets
    // ai: go), then filed into Downloads on the files' thread (fileTaken). asked: the root the poll asked for, whose ask
    // ai: ends here where it is not taken; a root the receiver holds verified with nothing to take is not asked for again
    // ai: while it holds it (skip). A take that fails leaves the file in the store and says why under the state line.
    private fun takeOut(asked: String? = null) {
        var took = ""
        var nothing = false
        try {
            val hh = h
            val f = (if (hh == 0L) null else parse(Native.rxStats(hh))?.get("file")) as? JsonObject
            val r = f?.str("root").orEmpty()
            if (f != null && f.flag("verified") && r.isNotEmpty()) {
                val path = if (Received.has(r)) "" else Native.rxFile(hh)
                if (path.isNotEmpty() && File(path).isFile) {
                    val dir = Received.take(File(path), f.str("type"), r)
                    took = r
                    post { filing += r }
                    later { fileTaken(dir, r) }
                } else nothing = r == asked
            }
        } catch (e: Exception) {
            val why = Received.whyOf(e)
            System.err.println("receive: the received file not taken out of the store: $why")
            nothing = asked != null
            post { if (asked != null) noteRoot = asked; fileNote = "Not saved: $why" }
        } finally {
            if (asked != null && asked != took) post { filing -= asked; if (nothing) skip = asked }
        }
    }
    // ai: a take filed (the files' thread): the list read again, its note, its filing ended
    private fun fileTaken(dir: File, r: String) {
        val res = runCatching { Received.file(dir) }
        val l = Received.list()
        post {
            files = l
            res.getOrNull()?.let { noted(listOf(it)) }
            res.exceptionOrNull()?.let { noteRoot = r; fileNote = "Not saved: ${Received.whyOf(it)}" }
            filing -= r
        }
    }
    // ai: the takes not in Downloads filed again (Received.retry), one pass queued at a time: at each look at the lists
    // ai: (Home, Receive entered; the first is the launch's, which files any a close cut short) and at each Start
    private var retrying = false
    private fun retryKept() {
        if (retrying) return
        retrying = true
        later {
            val done = Received.retry()
            val l = Received.list()
            post { retrying = false; files = l; noted(done) }
        }
    }
    // ai: What filings came to (the UI thread): the note set where the file the receiver holds, or one being filed, is not
    // ai: in Downloads, and cleared once the file it names is
    private var noteRoot = ""
    private fun noted(fs: List<Received.Filed>) {
        for (f in fs) {
            if (f.why.isEmpty()) { if (f.root == noteRoot) { noteRoot = ""; fileNote = "" } }
            else if (f.root == root || f.root in filing) { noteRoot = f.root; fileNote = "Not saved to Downloads: ${f.why}" }
        }
    }

    // ai: The stats, every quarter second while the session lives (rxStats: the receiver's, the camera's line inside)
    private val timer = Timer(250) { poll() }
    private var tick = 0
    private var endedRoot = ""                 // ai: the root the camera last stopped for
    private val filing = HashSet<String>()     // ai: roots asked to be taken out or being filed, until their filing ends
    private var skip = ""                      // ai: a root not asked for again while the receiver holds it (takeOut)
    private fun poll() {
        val hh = h
        if (hh == 0L) return
        val s = runCatching { Native.rxStats(hh) }.getOrDefault("")
        if (s.isEmpty()) return
        raw = s
        val j = parse(s)
        stats = j
        val r = rxOf(j)
        rx = r
        val f = j?.get("file") as? JsonObject
        root = f?.str("root").orEmpty()
        if (decoder == "auto" && running) j?.str("decoder")?.takeIf { it.isNotEmpty() }?.let { autoRan = if (it.startsWith("gpu")) "GPU" else "CPU" }
        if (phase == Phase.Loading && !configuring && r.state.isNotEmpty() && r.state != "starting" && r.state != "loading") {
            phase = Phase.On
            previewTimer.start()
        }
        val ended = (j?.get("camera") as? JsonObject)?.str("ended").orEmpty()
        // ai: a stream that ended by itself (a camera unplugged, a device error): the camera closed as well as stopped, so
        // ai: the next Start opens it again by its id
        if (ended.isNotEmpty() && (phase == Phase.On || phase == Phase.Loading)) { stop(); task { closeCam() }; phase = Phase.Error("The camera stopped: $ended") }
        // ai: the receiver let that root go (a clear, a new receiver): a copy of it received later is asked for anew
        if (skip.isNotEmpty() && skip != root) skip = ""
        if (r.verified && root.isNotEmpty()) {
            // ai: filed where it is in no list entry, no filing of it is under way and the session did not just find
            // ai: nothing to take (what is filed decides, as the app's keep, 2026-10-25: a file deleted at Home, or
            // ai: moved out of Downloads, and received again is filed again; the list read again where the file of its
            // ai: entry is gone)
            files.firstOrNull { it.root == root }?.let { if (!it.file.exists()) files = Received.list() }
            if (root !in filing && root != skip && files.none { it.root == root }) { val asked = root; filing += asked; task { takeOut(asked) } }
            // ai: the camera off once the file is in, once a root: the receiver holds the verified file after the stop,
            // ai: so a Start after it reads on
            if (root != endedRoot) { endedRoot = root; if (running) stop() }
        }
        replayTick(s)
        if (tick++ % 4 == 0 && app.screen == Screen.Home) thread(name = "lizard-partials", isDaemon = true) { val p = Partials.list(); post { partials = p } }
    }

    // ai: The preview: the newest frame's crop as the decoder reads it, at most PREVIEW px on its long side, grey, into
    // ai: one of two Skia bitmaps in turn (installPixels copies the bytes; Compose draws a bitmap by its copy), while On
    private val previewTimer = Timer(33) { pull() }
    private val buf = ByteArray(PREVIEW * PREVIEW)
    private var seq = 0L
    private val bitmaps = arrayOfNulls<Bitmap>(2)
    private var flip = 0
    private fun pull() {
        val hh = h
        if (hh == 0L || phase != Phase.On) return
        val got = runCatching { Native.rxPreview(hh, seq, PREVIEW, buf) }.getOrNull() ?: return
        if (got.size < 3) return
        val w = got[1].toInt(); val ht = got[2].toInt()
        if (w <= 0 || ht <= 0 || w.toLong() * ht > buf.size) return
        seq = got[0]
        flip = flip xor 1
        val bm = bitmaps[flip] ?: Bitmap().also { bitmaps[flip] = it }
        if (bm.installPixels(ImageInfo(w, ht, ColorType.GRAY_8, ColorAlphaType.OPAQUE), buf, w)) preview = bm.asComposeImageBitmap()
    }

    // ai: Save replays: a run while the switch is on and the camera reads (Native.data/replays/<run>, the app's run
    // ai: name, the newest REPLAY_FRAMES frames the decoder takes), ended when either stops: its rows (the stats a
    // ai: second, the newest REPLAY_ROWS) and the app's fields (camera, settings, build) into its meta. A run that ends
    // ai: with frames lets the finished runs before it go, as the app's; the folder is the replay.
    private val rows = ArrayDeque<String>()
    private var beginning = false
    private var replayRefused = false          // ai: a run the session would not begin: none asked again until the switch moves
    private var adopted = false
    private fun replayTick(s: String) {
        if (!adopted) adopt()
        val mine = runs.lastOrNull { it.state == RunState.Recording }
        val want = replays && phase == Phase.On
        if (want && mine == null && !beginning && !replayRefused) beginReplay()
        else if (!want && mine != null) endReplay(mine)
        if (mine != null && tick % 4 == 0) { rows.addLast(s); while (rows.size > REPLAY_ROWS) rows.removeFirst() }
    }
    private val replaysDir get() = File(Native.data, "replays")
    // ai: the runs on disk at the first look: the newest whole one (its meta.json written) listed, the rest deleted
    private fun adopt() {
        adopted = true
        val all = replaysDir.listFiles()?.filter { it.isDirectory }.orEmpty().sortedBy { it.name }
        val whole = all.lastOrNull { File(it, "meta.json").isFile }
        if (whole != null) {
            val meta = runCatching { parse(File(whole, "meta.json").readText()) }.getOrNull()
            val bytes = whole.listFiles()?.filter { it.name.endsWith(".gray") }?.sumOf { it.length() } ?: 0L
            runs.add(Run(whole.name, whole, 0L, RunState.Ready, meta?.num("frames")?.toLong() ?: 0L, bytes, meta?.str("error").orEmpty()))
        }
        val rest = all.filter { it != whole }
        if (rest.isNotEmpty()) thread(name = "lizard-replays", isDaemon = true) { rest.forEach { it.deleteRecursively() } }
    }
    private fun beginReplay() {
        beginning = true
        val fmt = SimpleDateFormat("yyyy-MM-dd'T'HH-mm-ss-SSS'Z'", Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }
        var t = System.currentTimeMillis()
        var run: String
        while (true) { run = "run-" + fmt.format(Date(t)); if (runs.none { it.run == run } && !File(replaysDir, run).exists()) break; t++ }
        val dir = File(replaysDir, run)
        rows.clear()
        task {
            val hh = h
            val r = if (hh == 0L) 0L else runCatching { replaysDir.mkdirs(); Native.rxReplayNew(dir.path, run, REPLAY_FRAMES) }.getOrDefault(0L)
            if (r != 0L) Native.rxRecord(hh, r)
            post {
                beginning = false
                if (r == 0L) replayRefused = true
                runs.add(Run(run, dir, r, if (r != 0L) RunState.Recording else RunState.Failed, why = if (r == 0L) "not begun" else ""))
                System.err.println("replay: $run" + if (r == 0L) " not begun" else "")
            }
        }
    }
    private fun putRun(r: Run) { val i = runs.indexOfFirst { it.run == r.run }; if (i >= 0) runs[i] = r }
    private fun endReplay(r: Run) {
        putRun(r.copy(state = RunState.Ending))
        val last = raw; val rowsNow = rows.joinToString("\n"); val more = moreJson()
        task {
            val hh = h
            if (hh != 0L) Native.rxRecord(hh, 0)
            // ai: on the files' thread, so a close waits for the run's meta.json (a run without one is deleted at the next look)
            later {
                val res = runCatching { Native.rxReplayEnd(r.handle, last, rowsNow, more) }
                post {
                    val j = res.getOrNull()?.let { parse(it) }
                    val n = j?.num("frames")?.toLong() ?: 0L
                    val done = if (n > 0) r.copy(state = RunState.Ready, frames = n, bytes = j?.num("bytes")?.toLong() ?: 0L, why = j?.str("error").orEmpty())
                        else r.copy(state = RunState.Failed, why = res.exceptionOrNull()?.message ?: j?.str("error").orEmpty().ifEmpty { "no frames" })
                    System.err.println("replay: ${r.run} " + if (done.state == RunState.Ready) "${done.frames} frames, ${done.bytes} B" else "failed: ${done.why}")
                    putRun(done)
                    val gone = if (done.state == RunState.Ready) runs.filter { it.run != r.run && (it.state == RunState.Ready || it.state == RunState.Failed) }
                        else listOf(done)
                    runs.removeAll(gone)
                    if (gone.isNotEmpty()) thread(name = "lizard-replays", isDaemon = true) { gone.forEach { it.dir.deleteRecursively() } }
                }
            }
        }
    }
    // ai: what the app adds to a run's meta.json: the camera, the settings, the build
    private fun moreJson(): String {
        val ci = info; val m = mode
        val cam = if (ci == null) JsonNull else JsonObject(mapOf("id" to JsonPrimitive(ci.id), "name" to JsonPrimitive(ci.name),
            "backend" to JsonPrimitive(ci.backend), "detail" to JsonPrimitive(ci.detail), "format" to JsonPrimitive(m?.format ?: ""),
            "size" to JsonPrimitive(m?.size ?: ""), "fps" to JsonPrimitive(m?.fps ?: 0.0)))
        return JsonObject(mapOf("camera" to cam, "zoom" to num(zoom), "exposure" to num(exposure), "gain" to num(gain), "layout" to JsonPrimitive(layout),
            "decoder" to JsonPrimitive(decoder), "batch" to JsonPrimitive(batch), "app" to JsonPrimitive(app.about))).toString()
    }
    private fun num(v: Double?): JsonElement = if (v == null || v.isNaN()) JsonNull else JsonPrimitive(v)

    // ai: What Receive shows
    val line: Readout.Line get() = Readout.line(on, phase == Phase.Starting, phase == Phase.Loading, (phase as? Phase.Error)?.why, rx,
        files.firstOrNull { it.root == root && it.bundle }?.files ?: 0)
    // ai: the lab line while the figures are this stream's (live), none otherwise (Readout.lab's while the camera is off)
    val lab: String get() = Readout.lab(live, rx)
    // ai: the preview's box: the crop's shape (1:1; 2:1 along the camera's long side)
    val cropRatio: Float get() = if (layout != "2:1") 1f else if ((mode?.h ?: 0) > (mode?.w ?: 1)) 0.5f else 2f
    // ai: One hint, the first that holds, while the camera runs and a word is read (2026-10-25): a camera, whose timing
    // ai: nothing steers, reads a sender at most half its rate without its captures spanning two pictures (the web's
    // ai: rule), so its rate, the one it agreed to at start where it names one (else the mode's), under twice the
    // ai: word's names the most the sender should paint; then a camera whose measured rate (the session's, from its
    // ai: fourth frame) is under the word's less one misses pictures.
    val hint: String get() {
        if (!on) return ""
        val j = stats ?: return ""
        // ai: a phone (camera/phone.cpp): its camera's rate (its frame count's, agreedFps) against what arrives, short
        // ai: where the link carries fewer (the phone skips evenly, 2026-10-27); a
        // ai: phone's camera is the phone's to time, so the rules after this are not its
        if (phone) {
            val cam = j["camera"] as? JsonObject
            val of = cam?.num("agreedFps") ?: 0.0
            val got = cam?.num("fps") ?: 0.0
            return if (of > 0 && got > 0 && got < 0.95 * of) "The link to the phone carries ${got.roundToInt()} of its ${of.roundToInt()} frames a second." else ""
        }
        val wf = (j["word"] as? JsonObject)?.num("fps") ?: 0.0
        if (rx.bandVersion <= 0 || wf <= 0) return ""
        val cam = j["camera"] as? JsonObject
        val camFps = cam?.num("agreedFps")?.takeIf { it > 0 } ?: mode?.fps ?: 0.0
        if (camFps > 0 && camFps < 2 * wf) {
            val n = floor(camFps / 2).toInt()
            if (n >= 1) return "Set the sender to at most $n fps for this camera."
        }
        val measured = cam?.num("fps") ?: 0.0
        if (measured > 0 && measured < wf - 1) return "This camera runs slower than the code changes, so some of it is missed."
        return ""
    }

    // ai: The readout (Developer Tools; the app's devReadout): a labelled line or two a part, from the stats (receiver.cpp
    // ai: stats, the session's camera line in "camera") and the camera's mode; a part with nothing to say left
    // ai: out. The receiver's figures of its last second (the decoder's ms, captured, decoded, found, the rate, the
    // ai: window's blocks) only while live: a camera that is off has none, and a stream's read 0 until its first second
    // ai: has closed; the held word, the totals and the file stand.
    val readout: String get() {
        val j = stats
        val now = live
        val out = mutableListOf<String>()
        fun put(label: String, vararg lines: String?) {
            lines.filterNotNull().filter { it.isNotEmpty() }.forEachIndexed { i, l -> out.add((if (i == 0) label else "").padEnd(9) + l) }
        }
        if (j != null) {
            val dec = j.str("decoder")
            put("decoder", if (dec.startsWith("gpu")) "GPU ${dec.removePrefix("gpu").trim()}, batches up to ${j.num("cap").toInt().takeIf { it > 0 } ?: j.num("B").toInt()}" +
                    (if (now) ", ${f(j.num("gpuMs"), 2)} ms of GPU a frame" else "")
                else if (dec == "cpu") "CPU, ${j.num("threads").toInt()} of ${j.num("threadsMax").toInt()} threads${if (j.flag("simd")) ", SIMD" else ""}" +
                    (if (now) ", ${f(j.num("cpuMs"), 1)} ms a frame" else "") else dec,
                j.str("gpuWhy").takeIf { dec == "cpu" && it.isNotEmpty() }?.let { "no GPU: $it" })
        }
        val cam = j?.get("camera") as? JsonObject
        val ci = info; val m = mode
        put("camera", if (ci != null && m != null && running) "${ci.name} (${ci.backend}): ${m.format} ${m.w} x ${m.h} at ${fpsText(m.fps)} fps, $layout crop" else "off",
            if (cam != null && running) "${f(cam.num("fps"), 1)} frames a second" + (if (now) ", ${f(j.num("capturedFps"), 0)} captured, " +
                "${f(j.num("processedFps"), 0)} decoded, ${j.num("dropped").toInt()} dropped a second" else "") else null,
            if (cam != null && running) "${cam.num("frames").toLong()} frames, ${cam.num("dropped").toLong()} dropped, ${cam.num("corrupt").toLong()} corrupt, " +
                "${f(cam.num("convertMs"), 2)} ms to luma" else null)
        if (j != null) {
            val word = j["word"] as? JsonObject
            val v = word?.num("version")?.toInt() ?: 0
            val named = if (v > 0) "LIZARD-${8 * v}${word?.num("fps")?.toInt()?.takeIf { it > 0 }?.let { " at $it fps" } ?: ""}" else "no word yet"
            val told = (j["totals"] as? JsonObject)?.let { t -> "${t.num("blocks").toLong()} blocks in ${t.num("frames").toLong()} frames all told" }
            if (now) put("code", "found ${f(100 * j.num("foundShare"), 0)}%" + (if (j.num("side") > 0) ", symbol ${f(j.num("side"), 0)} px" else "") + ", $named",
                "${Readout.rate(j.num("goodputKBs"))}, ${j.num("blocks").toInt()} blocks in ${f(j.num("windowSecs"), 2)} s" + (told?.let { "; $it" } ?: ""))
            else put("code", named, told)
            val file = j["file"] as? JsonObject
            put("file", when {
                j.flag("test") -> "the test stream"
                file == null -> "none yet"
                else -> "${file.str("name").ifEmpty { "(no name)" }}: ${Readout.partOf(file.num("received"), file.num("size").toLong())}" +
                    ", chunks ${file.num("chunksVerified").toInt()} of ${file.num("chunks").toInt()} verified${if (file.flag("verified")) ", whole" else ""}"
            })
        }
        j?.str("error")?.takeIf { it.isNotEmpty() }?.let { put("error", it) }
        return out.joinToString("\n")
    }

    companion object {
        const val PHONE_PORT = 47230   // ai: the phone's stream (liblizard/core/rx/stream.h TCP_PORT)
        const val PREVIEW = 720            // ai: the preview's long side at most, px
        const val REPLAY_FRAMES = 300      // ai: a run's frames (the web's and the app's 300)
        const val REPLAY_ROWS = 60         // ai: a run's stats rows, one a second
        // ai: the most the close waits for the files' thread (a large tar's unpack); what is left is done at the next launch
        const val CLOSE_FILING_SECS = 30L
        private fun f(x: Double, d: Int) = String.format(Locale.ROOT, "%.${d}f", x)
        // ai: a rate as a menu shows it: whole where it is (30, 60), else two decimals (29.97)
        fun fpsText(v: Double) = if (abs(v - v.roundToInt()) < 0.01) "${v.roundToInt()}" else String.format(Locale.ROOT, "%.2f", v)

        // ai: The stats' fields the screen reads (Readout.Rx; the app's MainActivity rxOf)
        fun rxOf(j: JsonObject?): Readout.Rx {
            if (j == null) return Readout.Rx()
            val f = j["file"] as? JsonObject
            return Readout.Rx(state = j.str("state"), error = j.str("error"), goodputKBs = j.num("goodputKBs"), foundShare = j.num("foundShare"),
                bandVersion = j.num("bandVersion").toInt(), fps = (j["word"] as? JsonObject)?.num("fps")?.toInt() ?: 0,
                hasFile = f != null, name = f?.str("name").orEmpty().ifEmpty { "a file" }, size = f?.num("size")?.toLong() ?: 0,
                received = f?.num("received")?.toLong() ?: 0, verified = f?.flag("verified") ?: false,
                sent = f?.num("sent")?.toLong() ?: 0, sentIn = f?.num("sentIn")?.toLong() ?: 0, secs = f?.num("secs") ?: 0.0)
        }
    }
}
