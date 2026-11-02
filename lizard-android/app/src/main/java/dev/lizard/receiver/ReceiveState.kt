package dev.lizard.receiver

import android.graphics.SurfaceTexture
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import dev.lizard.Session
import dev.lizard.pick.Pick
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.math.roundToInt
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.put
import dev.lizard.ui.*

// ai: The receiver (2026-10-31; Engine.kt, the app's Camera2 and its rules in Kotlin, before): the engine's receive
// ai: session (dev.lizard.Session over liblizard/core/rx/session.h, the desktop's too) made as Receive is entered and
// ai: let go once Receive is left and the camera is idle (letGo, as the desktop's: the session holds the data folder's
// ai: lock, so a window that is not receiving, a share's on Send, holds none), and what Receive shows of it. The
// ai: session runs every command in order on a thread of its own and owns what Engine and the poll did: the camera
// ai: (Camera2 through the NDK, liblizard/core/cam/android.cpp: the request, the zero-copy frames, the preview stream,
// ai: the autofocus on the code, the phase lock's delayed captures), the receiver made again on its transfer at a
// ai: decoder or crop change, the phase, a verified file taken into <data>/kept, the camera stopped once a file
// ai: is in, the camera's settings kept per camera (<data>/cameras.json: its mode, zoom a crop at a time), Save
// ai: replays, the one-session lock. The PC link is the process's, no session's (Session.link). Here: the session made
// ai: and let go, the start's order (the camera opened, its mode's preview size given to the TextureView's surface
// ai: before the stream starts, the surface handed over), the UI's state over its stats a quarter second
// ai: (MainActivity's poll), the camera's own rate against its peak (the heat warning's), the hints, the `rx:` log line
// ai: a second, the platform's fields into the stats and replays (the phone's heat, GPU clock, power, battery).
class ReceiveState(private val a: MainActivity) {
    // ai: what the camera was asked for and gave (its report in the stats: camera.report), for the crop's layout, the
    // ai: preview's turn and the Developer readout
    data class CamInfo(val id: String, val format: String, val size: Size, val preview: Size, val fps: Int, val note: String)
    // ai: a back camera the session lists (its id without the backend's "android:"), labelled as the menu shows it
    data class Lens(val id: String, val label: String)
    // ai: one of the camera's modes (session.h camera(): a resolution in a format, its preview stream's size)
    class Mode(val index: Int, val w: Int, val h: Int, val fps: Double, val format: String, val previewW: Int, val previewH: Int) {
        val size get() = "${w}x$h"
    }
    // ai: the camera's pipeline slowing (the heat warning's "Camera at N of 60 frames a second.")
    data class CameraSlow(val now: Int, val max: Int)

    val data = File(a.filesDir, "session")
    private val main = Handler(Looper.getMainLooper())
    // ai: the session made and let go, the start's order and the surface's hand-over, one at a time in the order asked
    // ai: (the hand-over waits for the camera; a session let go is closed here, so what comes after it finds it gone)
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "lizard-start") }
    private fun later(f: () -> Unit) = try { worker.execute(f) } catch (_: RejectedExecutionException) {}
    // ai: The session, and every call on it from any thread through use: a let-go waits for the calls under way (its
    // ai: handle freed after them), and a call after it finds none. Written on the start's thread alone.
    @Volatile private var session: Session? = null
    private val rw = ReentrantReadWriteLock()
    private fun <T> use(f: (Session) -> T): T? = rw.read { session?.let(f) }
    @Volatile private var closed = false
    // ai: the sessions made so far: a new one counts its takes from 0 (MainActivity's kept count)
    @Volatile var made = 0
        private set
    private var lettingGo = false      // ai: main's: a let-go queued and not yet done
    private var replaysOn = false      // ai: main's: Save replays as last asked, for a session made later

    var phase by mutableStateOf<RxPhase>(RxPhase.Idle)
        private set
    var cam by mutableStateOf<CamInfo?>(null)
        private set
    var lenses by mutableStateOf<List<Lens>>(emptyList())
        private set
    var modes by mutableStateOf<List<Mode>>(emptyList())
        private set
    var mode by mutableStateOf<Mode?>(null)           // ai: the camera's mode, running or next
        private set
    var zoomRange by mutableStateOf<ClosedFloatingPointRange<Float>?>(null)
        private set
    var zoom by mutableStateOf<Float?>(null)          // ai: the zoom as kept for the camera and crop, else its own
        private set
    var stats by mutableStateOf<JsonObject?>(null)    // ai: the newest stats as the session wrote them, read once a poll
        private set
    var link by mutableStateOf<JsonObject?>(null)     // ai: the PC link's state (session.h linkState)
        private set
    var cameraSlow by mutableStateOf<CameraSlow?>(null)
        private set
    var runs by mutableStateOf<List<Run>>(emptyList())
        private set
    var lock by mutableStateOf<JsonObject?>(null)     // ai: the phase lock's state, or {"off": why}
        private set
    var camId by mutableStateOf("")                   // ai: the camera open (no "android:")
        private set

    init {
        Native.load()
        listLenses()
    }

    // ai: the platform's fields for a replay's meta.json (the session adds the camera, its report, its settings, the lock's
    // ai: state and the receiver's)
    private fun replayMeta() = buildJsonObject { put("app", a.aboutLine()) }.toString()

    // ai: the back cameras the session lists, for Settings' Camera menu (off the main thread: each is read from the
    // ai: camera service)
    fun listLenses() = Thread({
        val l = parse(runCatching { Session.cameras() }.getOrDefault(""))?.objs("cameras").orEmpty()
            .filter { it.str("backend") == "android" }.map { Lens(it.str("id").removePrefix("android:"), it.str("name")) }
            .sortedBy { it.id.toIntOrNull() ?: Int.MAX_VALUE }
        main.post { lenses = l; Log.i(TAG, "back cameras: ${l.joinToString("; ") { it.label }}") }
    }, "lizard-cameras").start()

    // ai: The session where none is (the start's thread): made at once (its worker takes the data folder's lock, waiting
    // ai: for one of this process's still closing, adopts the replays and takes out what an earlier session left), the
    // ai: batch size and Save replays as asked; null where it could not be made, the phase saying why
    private fun session(): Session? {
        session?.let { return it }
        if (closed) return null
        return runCatching {
            Session(data.apply { mkdirs() }.path, Assets.dir(a).path, File(a.cacheDir, "lizard").apply { mkdirs() }.path, a.settings.decoder,
                a.settings.layout)
        }.onSuccess { s ->
            s.batchCap(a.settings.frames)
            s.replays(replaysOn, replayMeta())
            rw.write { session = s }
            made++
        }.onFailure { e ->
            Log.e(TAG, "session: ${e.message}")
            main.post { phaseTo(RxPhase.Error("The camera could not start: ${e.message}")) }
        }.getOrNull()
    }
    // ai: the start's thread: the session let go (the calls under way waited for, then its close: the camera stopped and
    // ai: closed, a recording replay written, a file verified since its last look taken into kept/, the lock released)
    private fun release() {
        val s = rw.write { session.also { session = null } } ?: return
        handed = null
        runCatching { s.close() }
    }

    // ai: The session let go off Receive (the desktop's letGo): off Receive, the camera idle (no start under way) and no
    // ai: replay recording or being written, at the poll. The transfer in hand stays in the store, which the next
    // ai: session resumes when its header is read again; a start meanwhile is queued behind the let-go and makes the next.
    private fun letGo() {
        if (session == null || lettingGo || a.screen == Screen.Receive || opening) return
        if (phase == RxPhase.On || phase == RxPhase.Starting || phase == RxPhase.Loading) return
        if (runs.any { it.state == RunState.Recording || it.state == RunState.Ending }) return
        lettingGo = true
        later { release(); main.post { lettingGo = false } }
    }

    // ai: The app's close: the session let go after what is queued, and the start's thread ended
    fun close() {
        closed = true
        gen++
        later { release() }
        worker.shutdown()
    }

    // ai: the camera
    @Volatile private var gen = 0      // ai: main's: each start and stop, so a start overtaken lets go
    private val startMu = Any()        // ai: a start's last look at gen and its call, against a stop's (stop)
    private var opening = false        // ai: main: a start under way before the session's own start (the phase shows it)
    private var chosen: Mode? = null   // ai: main: a mode chosen with no camera open to keep it: the next start names it

    // ai: Start: on the start's thread, the GPU tree in place (Assets) and the session made where none is, then asked, in
    // ai: order, for the decoder and crop chosen, the batch size, the camera closed and the one chosen opened (the
    // ai: closest-focusing back camera where none is or it is gone); the open waited for (the surface's hand-over is queued behind it and
    // ai: waits); its mode's preview size given to the surface (the size the camera service reads off it), the surface
    // ai: handed over (once the TextureView has made one, 2 s at most), and the stream started in the mode kept (or one
    // ai: Settings named).
    fun start() {
        if (closed) return
        val g = ++gen
        opening = true
        phaseTo(RxPhase.Starting)
        val st = a.settings
        val want = if (st.camera.isEmpty()) "" else "android:${st.camera}"
        val named = chosen
        chosen = null
        later {
            if (g != gen) return@later
            runCatching { Assets.ensure(a) }.onFailure { Log.e(TAG, "assets: ${it.message}") }
            val s = session()
            if (s == null) { main.post { if (g == gen) opening = false }; return@later }
            use { starting(it, g, st, want, named) }
        }
    }
    private fun starting(s: Session, g: Int, st: Settings, want: String, named: Mode?) {
        run {
            s.reconfigure(st.decoder, st.layout)
            s.batchCap(st.frames)
            // ai: every start opens the camera afresh (2026-11-01): a stop keeps it open and the session's open keeps an
            // ai: open camera of the same id, so a start after a stop went on in the same open; closed first, it opens again
            s.closeCamera()
            s.open(want)
            // ai: the open done: the session's commands run in order, and the hand-over of the surface it holds waits for
            // ai: its turn (nothing changes for the camera); then the camera's answer, or none where the open refused (the
            // ai: phase's error says why)
            s.previewWindow(handed)
            val cj = parse(runCatching { s.camera }.getOrDefault(""))?.takeIf { it.obj("info") != null }
            if (g != gen) return
            if (cj == null) { main.post { if (g == gen) opening = false }; return }
            val ms = modesOf(cj)
            val k = cj.obj("kept")?.let { if ("mode" in it) it.num("mode").toInt() else -1 } ?: -1
            val m = named?.let { n -> ms.firstOrNull { it.w == n.w && it.h == n.h && it.format == n.format } } ?: ms.getOrNull(k)
            if (m != null) sizePreview(m)
            // ai: the TextureView's surface, made by the view (2 s at most): the camera's second stream
            val t1 = SystemClock.elapsedRealtime()
            while (g == gen && texture == null && SystemClock.elapsedRealtime() - t1 < 2000) Thread.sleep(10)
            handWindow(s)
            // ai: the look at gen and the start one step against a stop's (startMu): a pause's stop queued between them
            // ai: would run before the start and leave the camera streaming behind a paused activity
            synchronized(startMu) {
                if (g != gen) return
                s.start(m?.takeIf { named != null }?.index?.toString() ?: "")
            }
            main.post { if (g == gen) opening = false }
        }
    }

    fun stop() {
        synchronized(startMu) {
            gen++
            use { it.stop() }
        }
        opening = false
        if (phase !is RxPhase.Error) phaseTo(RxPhase.Idle)
    }

    // ai: Settings' Camera: the camera chosen kept by the app (Settings.camera): started again where it reads, else opened
    // ai: (what it offers comes with the poll)
    fun cameraChosen(running: Boolean) {
        if (running) start() else use { it.open(if (a.settings.camera.isEmpty()) "" else "android:${a.settings.camera}") }
    }

    // ai: Settings' Resolution, Format and Frame rate: the camera's mode with that size in the format kept (else auto),
    // ai: the format at the size kept (a RAW format at any), or the rate at both; each at the rate kept where the lens
    // ai: offers it, else its fastest (the modes come fastest first): kept by the session as the camera's; the surface
    // ai: sized to its preview first, since a running stream starts again in it at once
    private fun atRate(ms: List<Mode>) = ms.firstOrNull { kotlin.math.abs(it.fps - (mode?.fps ?: 0.0)) < 0.01 } ?: ms.firstOrNull()
    fun chooseResolution(res: String) {
        val cur = mode
        val m = atRate(modes.filter { it.size == res && it.format == (cur?.format ?: "auto") }) ?: atRate(modes.filter { it.size == res && it.format == "auto" }) ?: return
        pickMode(m)
    }
    fun chooseFormat(fmt: String) {
        val cur = mode
        val m = atRate(modes.filter { it.format == fmt && it.size == (cur?.size ?: "1920x1080") }) ?: atRate(modes.filter { it.format == fmt }) ?: return
        pickMode(m)
    }
    fun chooseFps(f: Double) {
        val cur = mode ?: return
        val m = modes.firstOrNull { it.size == cur.size && it.format == cur.format && kotlin.math.abs(it.fps - f) < 0.01 } ?: return
        pickMode(m)
    }
    private fun pickMode(m: Mode) {
        mode = m
        if (session == null || camId.isEmpty()) { chosen = m; return }
        later { sizePreview(m); use { it.set("mode", m.index.toDouble()) } }
    }
    // ai: Settings' zoom: the camera moved live and the value kept for its crop (session.h set)
    fun zoomTo(z: Float) { zoom = z; use { it.set("zoom", z.toDouble()) } }
    // ai: another decoder or crop: a new receiver on the transfer in hand (the camera started again around it)
    fun reconfigure() { val st = a.settings; use { it.reconfigure(st.decoder, st.layout) } }
    fun batch(n: Int) { use { it.batchCap(n) } }
    fun replays(on: Boolean) { replaysOn = on; use { it.replays(on, replayMeta()) } }
    fun clear() { use { it.clear() } }
    // ai: an unfinished transfer deleted (Home's Delete): forgotten by the session where one is (the receiver's, or its
    // ai: folder), else its folder deleted off the main thread (session.h deletePartial: one of this process's sessions
    // ai: still closing waited for)
    fun deletePartial(p: Partial, done: () -> Unit) {
        if (use { it.forget(p.root.ifEmpty { p.id }) } != null) { main.postDelayed(done, 500); return }
        Thread({
            val why = Session.deletePartial(data.path, p.id)
            if (why.isNotEmpty()) Log.w(TAG, "partial: ${p.id} not deleted: $why")
            main.post(done)
        }, "lizard-partials").start()
    }

    // ai: The preview is a TextureView (a view like any other, so the page can clip it to the crop the decoder reads; a
    // ai: SurfaceView's layer is not clipped by its parents). Its stream's size is the SurfaceTexture's default buffer
    // ai: size when the camera's capture session is made (the camera service reads it off the surface), and a
    // ai: TextureView sets that to its own size whenever it is laid out (STATUS "The preview stretched"): so the mode's
    // ai: preview size is kept here and given again before every start (sizePreview) and at every reset by the view.
    // ai: The surface goes to the session (Session.previewWindow, waited for on the start's thread); one destroyed is
    // ai: kept until the camera no longer writes to it, then released.
    @Volatile private var texture: SurfaceTexture? = null
    @Volatile private var previewSize: Size? = null
    @Volatile private var surface: Surface? = null   // ai: main's: the surface made of the texture
    private var handed: Surface? = null    // ai: the start's thread: the surface the session holds
    private fun sizePreview(m: Mode) {
        if (m.previewW <= 0 || m.previewH <= 0) return
        val p = Size(m.previewW, m.previewH)
        previewSize = p
        texture?.setDefaultBufferSize(p.width, p.height)
    }
    // ai: the start's thread: the surface there now handed to the session, where it holds another or none
    private fun handWindow(s: Session) {
        val sf = surface
        if (sf == handed || sf == null || !sf.isValid) return
        val why = s.previewWindow(sf)
        if (why.isEmpty()) handed = sf else Log.w(TAG, "preview: $why")
    }
    val textureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, hh: Int) {
            texture = st
            previewSize?.let { st.setDefaultBufferSize(it.width, it.height) }
            surface = Surface(st)
            later { use { handWindow(it) } }
        }
        override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, hh: Int) {
            previewSize?.let { st.setDefaultBufferSize(it.width, it.height) }
        }
        override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
            texture = null
            val sf = surface
            surface = null
            try {
                worker.execute {
                    if (handed == sf) { use { it.previewWindow(null) }; handed = null }
                    main.post { sf?.release(); st.release() }
                }
            } catch (_: RejectedExecutionException) { sf?.release(); st.release() }
            return false
        }
        override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
    }

    // ai: the PC link: the process's, whether a session lives or not (session.h link); an accessory's fd the bridge's,
    // ai: which closes it where one is up already
    fun linkUp(layout: String) { Session.link(true, layout) }
    fun linkDown() { Session.link(false, "") }
    fun linkPack(on: Boolean) { Session.linkPack(on) }
    fun accessoryOpen(): Boolean = parse(runCatching { Session.linkState() }.getOrDefault(""))?.flag("accessory") == true
    fun usbAccessory(fd: Int): Boolean = Session.usbAccessory(fd).isEmpty()

    // ai: the poll
    private var camJson = ""
    private var keptFor = ""
    private var tick = 0
    private var resAsked = ""

    private fun phaseTo(p: RxPhase) {
        if (p == phase) return
        phase = p
        a.onPhase(p)
    }

    // ai: Every quarter second (MainActivity's poll): the stats, the camera and the PC link read off the main thread (each
    // ai: JSON parsed once), the UI's state from them here, the session let go where it may be (letGo); the stats, null
    // ai: with no session
    suspend fun poll(platform: String?): JsonObject? {
        val (st, cj, lk) = withContext(Dispatchers.Default) {
            val lk = runCatching { Session.linkState() }.getOrDefault("")
            use { s ->
                if (platform != null) runCatching { s.extra(platform) }
                Triple(runCatching { s.stats }.getOrDefault(""), runCatching { s.camera }.getOrDefault(""), lk)
            } ?: Triple("", "", lk)
        }
        link = parse(lk)
        val j = parse(st)
        stats = j
        if (j == null) return null
        lock = j.obj("lock")
        // ai: the phase: the session's, Starting while the start's own order runs (before the session's start)
        phaseTo(rxPhaseOf(j, opening))
        val c = j.obj("camera")
        c?.obj("report")?.let { r -> camInfoOf(c.str("id").removePrefix("android:"), r)?.let { if (it != cam) cam = it } }
        cameraTick(cj)
        if (tick++ % 4 == 0) {
            slowTick(c)
            rxLine(j)
            // ai: test: `adb shell setprop debug.lizard.resolution <WxH>` picks that resolution as the panel's menu does,
            // ai: acted on when the value changes, so the panel's own choice stands until the property is set again
            val res = Native.prop("debug.lizard.resolution")
            if (res != resAsked) { resAsked = res; if (res.matches(Regex("\\d+x\\d+")) && res != mode?.size) { Log.i(TAG, "resolution: $res asked"); chooseResolution(res) } }
        }
        runsOf(j)?.let { if (it != runs) runs = it }
        letGo()
        return j
    }

    private fun camInfoOf(id: String, r: JsonObject): CamInfo? {
        fun size(k: String) = r.str(k).split("x").mapNotNull { it.toIntOrNull() }.takeIf { it.size == 2 }?.let { Size(it[0], it[1]) }
        val sz = size("size")?.takeIf { it.width > 0 } ?: return null
        val fps = r.str("fps").trim('[', ']').split(",").mapNotNull { it.trim().toIntOrNull() }.lastOrNull() ?: 0
        return CamInfo(id, r.str("format"), sz, size("preview") ?: sz, fps, r.str("note"))
    }

    private fun modesOf(j: JsonObject): List<Mode> = (j["modes"] as? JsonArray)?.mapIndexedNotNull { i, e ->
        (e as? JsonObject)?.let { o ->
            val p = (o["preview"] as? JsonArray)?.map { ((it as? JsonPrimitive)?.doubleOrNull ?: 0.0).toInt() }
            Mode(i, o.num("w").toInt(), o.num("h").toInt(), o.num("fps"), o.str("format"), p?.getOrNull(0) ?: 0, p?.getOrNull(1) ?: 0)
        }
    }.orEmpty()

    // ai: the camera open as the session describes it: Settings' modes and the mode running or next; its zoom as kept
    // ai: for it in this crop, taken up when it opens or its crop changes (not while the bar moves)
    private fun cameraTick(cj: String) {
        if (cj == camJson) return
        camJson = cj
        val j = parse(cj) ?: return
        val info = j.obj("info") ?: return
        camId = info.str("id").removePrefix("android:")
        val ms = modesOf(j)
        modes = ms
        val started = if ("mode" in j) j.num("mode").toInt() else -1
        val kept = j.obj("kept")
        mode = ms.getOrNull(if (started >= 0) started else kept?.let { if ("mode" in it) it.num("mode").toInt() else -1 } ?: -1) ?: mode
        val z = j.objs("controls")?.firstOrNull { it.str("id") == "zoom" }
        zoomRange = z?.let { it.num("min").toFloat()..maxOf(it.num("min").toFloat() + 0.1f, it.num("max").toFloat()) }
        val key = "$camId/${a.settings.layout}"
        if (key != keptFor) {
            keptFor = key
            zoom = (kept?.get("zoom") as? JsonPrimitive)?.doubleOrNull?.toFloat() ?: z?.num("value")?.toFloat()
        }
    }

    // ai: The camera's pipeline slowing (2026-10-07): once its rate a second (the session's measure) has stood within 5% of
    // ai: the rate asked, a second under 90% of that peak, twice running, is the camera completing fewer frames (the phone
    // ai: warm), and the heat warning says so (heatText, Clocks.kt); back at 95% of the peak twice running clears it
    private var peak = 0.0
    private var slowFor = 0
    private var okFor = 0
    private fun slowTick(c: JsonObject?) {
        if (c == null || !c.flag("streaming")) { peak = 0.0; slowFor = 0; okFor = 0; if (cameraSlow != null) cameraSlow = null; return }
        val rate = c.num("fps")
        val max = c.num("agreedFps").roundToInt()
        if (rate <= 0 || max <= 0) return
        if (rate > peak) peak = rate
        val slow = cameraSlow
        if (peak >= 0.95 * max && rate < 0.9 * peak) {
            okFor = 0
            if (++slowFor >= 2 && slow?.now != rate.roundToInt()) {
                if (slow == null) Log.i(TAG, "camera: slowed to %d of %d a second (peak %.1f)".format(rate.roundToInt(), max, peak))
                cameraSlow = CameraSlow(rate.roundToInt(), max)
            }
        } else {
            slowFor = 0
            if (slow != null && rate >= 0.95 * peak && ++okFor >= 2) { Log.i(TAG, "camera: back at its rate (peak %.1f a second)".format(peak)); cameraSlow = null }
        }
    }

    // ai: One hint, the first that holds, while the camera reads a word (dev.lizard.ui hintOf; the desktop's rules): a
    // ai: camera the phase lock holds takes no rate hint
    fun hint(j: JsonObject?): String = hintOf(j, phase == RxPhase.On, locked = lock?.let { "off" !in it } == true)


    // ai: A second's line for the log while a symbol is in view (Engine.rx before; tools/phone read it): what the receiver
    // ai: read, and what it could have read at this format. With the phase held every capture is a clean one, so the
    // ai: ceiling is a clean capture's blocks at the painted rate, read off the captures since the last line, whatever
    // ai: share of them the phase let through: clean are those with at least half the blocks of the third best of them
    // ai: (and an eighth of the word's), the ceiling their mean, 472 B a block, at the painted rate or the camera's if
    // ai: that is lower. At the end: the clean captures that read no new block (a sender or compositor late), the frames
    // ai: the receiver dropped, and the decode's ms a frame. The tools' patterns stop before them.
    private var rxSeen = 0.0
    private fun rxLine(j: JsonObject) {
        val blocks = ArrayList<Int>()
        val fresh = ArrayList<Int>()
        var newest = rxSeen; var first = Double.MAX_VALUE; var all = 0
        for (e in (j["series"] as? JsonArray).orEmpty()) {
            val f = (e as? JsonArray)?.map { (it as? JsonPrimitive)?.doubleOrNull ?: Double.NaN } ?: continue
            val ms = f.getOrNull(0) ?: continue
            if (!(ms > rxSeen)) continue
            all++
            if (ms > newest) newest = ms
            if (ms < first) first = ms
            if (f.getOrNull(3) == 1.0) { blocks.add((f.getOrNull(1) ?: 0.0).toInt()); fresh.add((f.getOrNull(2) ?: 0.0).toInt()) }
        }
        rxSeen = newest
        val w = j.obj("word")
        val version = w?.num("version")?.toInt() ?: 0
        val painted = w?.num("fps")?.toInt() ?: 0
        if (version == 0 || blocks.size < 8 || 2 * blocks.size < all) return
        val byBlocks = blocks.sortedDescending()
        val least = maxOf(1, (byBlocks[minOf(2, byBlocks.size - 1)] + 1) / 2, (Native.blocksFor(Pick.STEP * version) + 7) / 8)
        val clean = blocks.filter { it >= least }
        val repeated = blocks.indices.count { blocks[it] >= least && fresh[it] == 0 }
        val gpuMs = j.num("gpuMs")
        val cost = if (gpuMs > 0) "%.2f ms of GPU a frame".format(gpuMs) else "%.1f ms of CPU a frame".format(j.num("cpuMs"))
        val each = if (clean.isEmpty()) 0.0 else clean.average()
        val camera = if (all > 1 && newest > first) (all - 1) * 1000.0 / (newest - first) else 0.0
        val rate = if (painted > 0) minOf(painted.toDouble(), camera) else camera
        // ai: the symbol's side in camera px against the crop each frame gives it (2:1: half the frame's long side, at most
        // ai: its short side), and the share of frames found
        val crop = cam?.size?.let { s -> if (a.settings.layout == "2:1") minOf(maxOf(s.width, s.height) / 2, minOf(s.width, s.height)) else minOf(s.width, s.height) } ?: 0
        Log.i(TAG, "rx: %s, LIZARD-%d painted %d, goodput %.0f KB/s, %d of %d captures clean, %.1f blocks a clean capture, ceiling %.0f KB/s, %d frames a second, symbol %.0f px of a %d crop, found %.0f%%, %d repeated, %d dropped, %s".format(
            j.str("state"), Pick.STEP * version, painted, j.num("goodputKBs"), clean.size, blocks.size, each, each * Pick.PAYLOAD * rate / 1000, j.num("processedFps").toInt(),
            j.num("side"), crop, 100 * j.num("foundShare"), repeated, j.num("dropped").toInt(), cost))
    }

    companion object {
        // ai: the session's unfinished transfers (session.h partials), Home's Unfinished
        fun partialsOf(dataDir: File): List<Partial> = dev.lizard.ui.partialsOf(Session.partials(dataDir.path))
    }
}
