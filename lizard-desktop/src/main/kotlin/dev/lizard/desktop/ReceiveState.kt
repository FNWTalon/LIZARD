package dev.lizard.desktop

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asComposeImageBitmap
import dev.lizard.Session
import dev.lizard.pick.Pick
import dev.lizard.readout.Readout
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import org.jetbrains.skia.Bitmap
import org.jetbrains.skia.ColorAlphaType
import org.jetbrains.skia.ColorType
import org.jetbrains.skia.ImageInfo
import java.awt.EventQueue
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import javax.swing.Timer
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.roundToInt
import dev.lizard.ui.*

// ai: The receiver (2026-10-25, the PC receiver; the Android app's Engine and MainActivity, the web's recv.mjs): the
// ai: engine's receive session (dev.lizard.Session over liblizard/core/rx/session.h, since 2026-10-31) made at the first
// ai: start, and what Receive shows of it. The session runs every command in order on a thread of its own and owns the
// ai: rules a poll applied here before: the phase (Idle, Starting, Loading, On, Error with its sentence, the stats'
// ai: "phase" and "why"), a verified file taken out of the store into Native.data/kept (the stats' "kept"), the camera
// ai: stopped once a file is in, a stream that ended closed, a phone's own Start and Stop followed, the camera's
// ai: settings kept per camera (Native.data/cameras.json: its mode, zoom a crop at a time, exposure and gain), Save
// ai: replays, the one-session lock. Here: the UI's state over its stats a quarter second and its preview about 60 times a
// ai: second (stats, preview: neither waits), the cameras listed (a thread of their own: a listing opens devices), the
// ai: takes filed into Downloads (Received.kt, on the files' thread), the hint and the readout. The session and any
// ai: transfer partly in outlive a stop and the screen: leaving Receive stops the camera and closes it (another app may
// ai: want it); off Receive and idle the session is let go (letGo: its lock with it); the app's close ends it (the
// ai: camera's controls put back, a recording replay written). The figures of the receiver's last second show only while
// ai: they are this stream's (the stats' "live"): none while the camera is off, a received file's line kept.
class ReceiveState(private val app: App) {
    // ai: what the session says of the cameras (Session.cameras, camera): a camera listed, a mode, a control
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

    var phase by mutableStateOf<RxPhase>(RxPhase.Idle)
    var cameras by mutableStateOf<List<Cam>>(emptyList())
    var camera by mutableStateOf(Prefs.Rx.camera)              // ai: the camera chosen ("": the first listed)
    var info by mutableStateOf<CamInfo?>(null)                 // ai: the camera open
    // ai: Windows: the phones and USB3 Vision cameras with no driver libusb can use, and a LIZARD driver's presence
    // ai: (UsbDriver); the one chosen in the Camera menu, Install USB driver's; its run, and what it said where it failed
    var usb by mutableStateOf<UsbDriver.State?>(null)
    var driverFor by mutableStateOf<UsbDriver.Device?>(null)
    var driverBusy by mutableStateOf(false)
    var driverNote by mutableStateOf("")
    private var pickUsbPhone = false
    private var pickCamera = false
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
    var experimentalOpen by mutableStateOf(Prefs.Rx.experimentalOpen)
    var collapsed by mutableStateOf(Prefs.Rx.collapsed)
    var showStats by mutableStateOf(Prefs.Rx.stats)
    // ai: Save replays: kept across starts, as the Android app's (Settings.replays)
    var replays by mutableStateOf(Prefs.Rx.replays)
    var files by mutableStateOf<List<Received.Entry>>(emptyList())
    // ai: why the newest file is not in Downloads (Received.Filed's why; it is in the app's own folder then, listed from
    // ai: there), "" where it is; shown under the state line until a later filing of it puts it there
    var fileNote by mutableStateOf("")
    var partials by mutableStateOf<List<Partial>>(emptyList())
    val runs = mutableStateListOf<Run>()

    // ai: the test hook's (ReceiverTest.kt): a camera that must be listed, and a mode by index or "WxH@fps"
    var testCamera: String? = null
    var testMode: String? = null

    val running get() = phase == RxPhase.Starting || phase == RxPhase.Loading || phase == RxPhase.On
    val on get() = phase == RxPhase.On
    // ai: the receiver's figures of its last second are this stream's (session.h stats "live"): false while the camera
    // ai: is off; from a start the receiver's own, 0 until the stream's first second has closed
    val live get() = on && stats?.flag("live") == true
    val testing get() = on && rx.state == "test"
    // ai: the receiver holds a verified file (the camera stops then); none while the test stream is read
    val holds get() = rx.verified && !testing && root.isNotEmpty()
    // ai: that file's entry, once it is in the list (filed into Downloads a moment after the camera stops)
    val kept: Received.Entry? get() = if (holds) files.firstOrNull { it.root == root } else null
    val cpu get() = decoder == "cpu" || (decoder == "auto" && autoRan == "CPU")

    // ai: The received files' own work, one at a time in the order asked: a filing pass (Received.retry: each take in
    // ai: kept/ into Downloads, a tar unpacked), Home's deletes, an unfinished transfer's folder deleted with no session.
    // ai: close() waits for it: the app's exit, once the window is gone, cut each short before (a half-unpacked bundle in
    // ai: Downloads, an entry delisted with its file left).
    private val filer = Executors.newSingleThreadExecutor { r -> Thread(r, "lizard-files").apply { isDaemon = true } }
    private var session: Session? = null       // ai: UI thread: the session, made at the first start
    private var closed = false

    private fun post(f: () -> Unit) = EventQueue.invokeLater(f)
    // ai: on the files' thread; once the close has let it go, said and not done (a take not filed is filed at the next
    // ai: launch, Received.retry)
    private fun later(f: () -> Unit) {
        try {
            filer.execute { try { f() } catch (e: Throwable) { System.err.println("receive: ${e.javaClass.simpleName}: ${e.message}") } }
        } catch (_: RejectedExecutionException) { System.err.println("receive: the app is closing: left for the next launch") }
    }

    // ai: the platform's fields for a replay's meta.json (the session adds the camera, its settings and the receiver's)
    private fun replayMeta() = JsonObject(mapOf("app" to JsonPrimitive(app.about))).toString()

    // ai: The session, made where none is (at once: its receiver is made at its first start, its lock taken on its own
    // ai: thread, the takes an earlier session left taken out there); the cameras' settings this app kept before the
    // ai: session kept them moved into its cameras.json first (moved, waited for: its file is read as the session is
    // ai: made). Null where it could not be made (the phase says why).
    private fun session(): Session? {
        session?.let { return it }
        if (closed) return null
        val m = moving()
        m.join(MOVE_WAIT_MS)
        synchronized(moveMu) { if (m.isAlive) moveGivenUp = true }
        return try {
            val cache = File(Native.buildDir, "pipeline").apply { mkdirs() }
            Session(Native.data.apply { mkdirs() }.path, Native.assets, cache.path, decoder, layout, forgetSettings = Prefs.forgotten).also { s ->
                session = s
                taken = -1L   // ai: a new session counts its takes from 0
                seq = 0L; preview = null   // ai: and its frames from 0: the last session's picture is not this one's
                s.batchCap(batch)
                s.replays(replays, replayMeta())
                timer.start()
            }
        } catch (e: Exception) {
            phase = RxPhase.Error("The camera did not start: ${e.message ?: e.javaClass.simpleName}")
            null
        }
    }
    // ai: The cameras' settings this app kept before the session (Prefs.Rx.camerasMoved) into cameras.json once, where it
    // ai: is not yet, on a thread of its own begun at the first look at Home's lists (the native library loaded by then)
    // ai: or the first session: a camera kept under its id's hash is matched against the cameras listed, a listing
    // ai: opening devices (seconds on some systems), done only where such a key is. A move not done by MOVE_WAIT_MS (a
    // ai: listing that hangs) writes nothing (moveGivenUp), so no session's file is written under it.
    private val moveMu = Any()
    private var moveGivenUp = false      // ai: moveMu
    private var moved: Thread? = null    // ai: UI thread
    private fun moving(): Thread = moved ?: thread(name = "lizard-camera-move", isDaemon = true) { moveCameraSettings() }.also { moved = it }
    private fun moveCameraSettings() {
        val f = File(Native.data, "cameras.json")
        if (f.exists()) return
        val j = Prefs.Rx.camerasMoved { parse(Session.cameras())?.objs("cameras").orEmpty().map { it.str("id") } } ?: return
        synchronized(moveMu) {
            if (moveGivenUp) return
            writeCameras(f, j)
        }
    }
    private fun writeCameras(f: File, j: String) {
        runCatching {
            Native.data.mkdirs()
            val tmp = File(Native.data, "cameras.json.tmp")
            tmp.writeText(j)
            java.nio.file.Files.move(tmp.toPath(), f.toPath(), java.nio.file.StandardCopyOption.ATOMIC_MOVE)
        }.onFailure { System.err.println("receive: the cameras' settings not moved into ${f.path}: ${it.message}") }
    }

    // ai: the cameras every backend sees, labelled for the menu: the device's name, a number after a name two share
    private fun listed(): List<Cam> {
        val j = parse(Session.cameras()) ?: return emptyList()
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
    // ai: a phone's crop, from its listing's detail ("2:1, <ip>"; cam/phone.cpp), null for any other camera
    private fun phoneLayout(cams: List<Cam>, id: String) =
        cams.firstOrNull { it.id == id && it.backend == "phone" }?.detail?.substringBefore(",")?.trim()?.takeIf { it == "1:1" || it == "2:1" }
    val phone get() = info?.backend == "phone" || camera.startsWith("phone:")
    // ai: The Upload camera (2026-10-28, cam/upload.cpp): a .lizard (the phone's Download) or a replay's folder played
    // ai: as a camera, upload:<path>, asked for by the Camera menu's Upload (Chooser); never listed, never kept as the
    // ai: camera chosen, its crop the file's (its detail, as a phone's), its end the stream's
    val upload get() = info?.backend == "upload" || camera.startsWith("upload:")
    // ai: frames that are another device's crop at its own rate (a phone, an upload): no modes, crop or controls here
    val fixed get() = phone || upload
    val uploaded: String? get() = camera.takeIf { it.startsWith("upload:") }
    fun chooseUpload() {
        val owner = app.frame ?: return
        Chooser.choose(owner, "Choose a .lizard file", false, false) { f -> f.firstOrNull()?.let { chooseCamera("upload:" + it.path) } }
    }
    // ai: A phone by its address (2026-10-27, beside the beacon and the gateways, which a firewall or a route may defeat): the
    // ai: Camera menu's last entry asks for it (askAddress); the address typed is the camera's id, phone:<ip>:<port>
    // ai: (the stream's port where none is typed), listed by it while no listing has it
    var askAddress by mutableStateOf(false)
    val typedPhone: String? get() = camera.takeIf { c -> c.startsWith("phone:") && !c.startsWith("phone:usb:") && cameras.none { it.id == c } }
    // ai: the cameras listed, and the one open while it is not (none is listed until the Camera menu first opens: the field
    // ai: names the camera open, not its blank "Camera", 2026-11-02)
    val cameraOptions: List<Pair<String, String>> get() =
        cameras.map { it.id to it.label } + listOfNotNull(info?.takeIf { i -> cameras.none { it.id == i.id } }?.let { it.id to it.name }) + usb?.devices.orEmpty().map { it.id to "${it.name} (no USB driver)" } + listOfNotNull(typedPhone?.let { it to "Phone at ${it.removePrefix("phone:").removeSuffix(":$PHONE_PORT")}" }) +
            listOfNotNull(uploaded?.let { it to File(it.removePrefix("upload:")).name }) + ("phone:" to "Phone by address") + ("upload:" to "Upload")
    fun chooseCameraOption(id: String) {
        askAddress = id == "phone:"
        driverFor = usb?.devices?.firstOrNull { it.id == id }
        if (driverFor != null) return
        if (id == "upload:") chooseUpload() else if (id != "phone:") chooseCamera(id)
    }
    fun choosePhoneAddress(typed: String) {
        val a = typed.trim()
        if (!Regex("\\d{1,3}(\\.\\d{1,3}){3}(:\\d{1,5})?").matches(a)) return
        askAddress = false
        chooseCamera("phone:" + if (':' in a) a else "$a:$PHONE_PORT")
    }

    private fun camInfo(j: JsonObject): CamInfo? {
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
    // ai: the camera open as the session describes it now (its controls' values as last read; never waits), "{}" with
    // ai: none (the test hook's last line)
    fun cameraNow(): String = session?.let { runCatching { it.camera }.getOrNull() } ?: "{}"

    // ai: a mode chosen in Settings with no session to keep it (Resolution, Format, Frame rate): the next start names it
    private var chosenMode: Mode? = null

    // ai: Start: the session (made where none is) asked, in order, for the decoder and crop chosen (a
    // ai: phone's or an upload's own crop is the session's), the batch size, the camera chosen (the first listed where
    // ai: it is none or gone), and its stream in its kept mode (or one the test hook or Settings named); the phase is
    // ai: the session's from the poll right after.
    fun start() {
        if (app.loadError.isNotEmpty()) { phase = RxPhase.Error(app.loadError); return }
        if (running) return
        // ai: a Start tries again what did not go: the files Downloads did not take
        retryKept()
        val s = session() ?: return
        val id = testCamera ?: camera
        s.reconfigure(decoder, layout)
        s.batchCap(batch)
        // ai: every start opens the camera afresh (2026-11-01): a Stop keeps it open and the session's open keeps an open
        // ai: camera of the same id, so a Start after a Stop went on in the same open; closed first, it opens again
        s.closeCamera()
        s.open(id)
        val named = testMode ?: chosenMode?.takeIf { info?.id == id }?.index?.toString() ?: ""
        chosenMode = null
        s.start(named)
        poll()
    }
    fun stop() {
        session?.stop() ?: run { if (running) phase = RxPhase.Idle }
        poll()
    }
    fun toggle() = if (running) stop() else start()
    // ai: Receive entered (App.go): the lists read again, the camera started
    fun enter() { refreshLists(); start(); stamp = null; watchTimer.start() }
    // ai: Receive left: the camera stopped and closed; the session and its transfer kept
    fun leave() { watchTimer.stop(); stop(); session?.closeCamera() }

    // ai: The app's close: the session ended on a thread of its own, waited for a few seconds at most (its camera
    // ai: stopped and closed, its controls put back, a recording replay written, a file verified since its last look
    // ai: taken out into kept/); then the files' thread: the filing under way and one more pass for what the session took
    // ai: last, CLOSE_FILING_SECS at most (the window is hidden by then, App.close); a filing still under way after that
    // ai: is done again from the app's folder at the next launch (Received.retry), its half-made unpack never in place.
    fun close() {
        timer.stop(); watchTimer.stop()
        closed = true
        val s = session
        session = null
        closing?.join(8000)
        if (s != null) thread(name = "lizard-rx-close", isDaemon = true) { runCatching { s.close() } }.join(8000)
        later { Received.retry() }
        filer.shutdown()
        if (!filer.awaitTermination(CLOSE_FILING_SECS, TimeUnit.SECONDS)) System.err.println("receive: a filing still under way at the close: done at the next launch")
    }

    // ai: Settings
    fun chooseCamera(id: String) {
        if (id == camera && info?.id == id) return
        camera = id
        if (!id.startsWith("upload:")) Prefs.Rx.camera = id
        layout = phoneLayout(cameras, id) ?: Prefs.Rx.layout   // ai: a phone's crop its own; another camera's the one kept
        zoom = null; exposure = null; gain = null
        chosenMode = null
        if (running) { stop(); start() }
        else session?.open(id)   // ai: what it offers comes with the poll; a refusal is the phase's
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
        pickMode(m)
    }
    fun chooseFormat(format: String) {
        val ci = info ?: return
        val cur = mode ?: return
        val at = ci.modes.filter { it.size == cur.size && it.format == format }
        val m = at.firstOrNull { abs(it.fps - cur.fps) < 0.01 } ?: at.maxByOrNull { it.fps } ?: return
        pickMode(m)
    }
    fun chooseFps(f: Double) {
        val ci = info ?: return
        val cur = mode ?: return
        val at = ci.modes.filter { it.size == cur.size && abs(it.fps - f) < 0.01 }
        val m = at.firstOrNull { it.format == cur.format } ?: at.minByOrNull { if (it.compressed) 1 else 0 } ?: return
        pickMode(m)
    }
    // ai: the camera's mode: kept by the session as the camera's (its "mode" setting); running, the stream started again
    // ai: in it (a stop and a start, as a hand would); with no session, named at the next start
    private var pendingMode: Mode? = null   // ai: the mode pickMode started the camera again in, until it reports it
    private var pendingStopped = false      // ai: and whether the camera has shown it stopped since
    private fun pickMode(m: Mode) {
        mode = m
        if (running) { chosenMode = m; pendingMode = m; pendingStopped = false; stop(); start() }
        else session?.set("mode", m.index.toDouble()) ?: run { chosenMode = m }
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
        reconfigure()   // ai: the crop's own zoom (a crop keeps one of its own) the session's at the new receiver
    }
    // ai: a new receiver for the decoder and crop chosen, on the transfer in hand (the session stops the camera and
    // ai: starts it again around it; until its stats are its own the phase is Loading)
    private fun reconfigure() {
        val s = session ?: return
        s.reconfigure(decoder, layout)
        poll()
    }
    fun zoomTo(v: Double) { zoom = v; session?.set("zoom", v) }
    // ai: null: the camera's own exposure (automatic where it can be)
    fun exposureTo(us: Double?) { exposure = us; session?.set("exposure", us ?: 0.0, us == null) }
    fun gainTo(v: Double?) { gain = v; session?.set("gain", v ?: 0.0, v == null) }
    fun chooseBatch(n: Int) { batch = n.coerceIn(1, 32); Prefs.Rx.batch = batch }
    fun commitBatch() { session?.batchCap(batch) }
    fun toggleSettings() { settingsOpen = !settingsOpen; Prefs.Rx.settingsOpen = settingsOpen }
    fun toggleTools() { toolsOpen = !toolsOpen; Prefs.Rx.toolsOpen = toolsOpen }
    fun toggleExperimental() { experimentalOpen = !experimentalOpen; Prefs.Rx.experimentalOpen = experimentalOpen }
    fun toggleRail() { collapsed = !collapsed; Prefs.Rx.collapsed = collapsed }
    fun chooseStats(on: Boolean) { showStats = on; Prefs.Rx.stats = on }
    fun chooseReplays(on: Boolean) { replays = on; Prefs.Rx.replays = on; session?.replays(on, replayMeta()) }

    // ai: the cameras read again (the Camera menu opening, the camera watch's stamp moved), off the UI thread: a backend's
    // ai: listing opens each device (V4L2's nodes, Media Foundation's sources)
    private var listing = false
    fun refreshCameras() {
        if (listing || app.loadError.isNotEmpty()) return
        listing = true
        thread(name = "lizard-cameras", isDaemon = true) {
            val l = runCatching { listed() }.getOrNull()
            val u = if (UsbDriver.here) UsbDriver.state() else null
            post {
                listing = false
                usb = u
                if (driverFor != null && u?.devices?.none { it.id == driverFor?.id } != false) driverFor = null
                if (l != null) {
                    cameras = l
                    dropGonePhone()
                    // ai: the phone or camera its driver was installed for, chosen once it is listed
                    if (pickUsbPhone) l.firstOrNull { it.id.startsWith(if (pickCamera) "aravis:" else "phone:usb:") }?.let { pickUsbPhone = false; chooseCamera(it.id) }
                }
            }
        }
    }

    // ai: Install USB driver (Windows, UsbDriver): WinUSB for the device chosen (a phone's accessory too), then the
    // ai: cameras listed again and the device chosen as it shows; Remove USB driver: every LIZARD driver off, each
    // ai: device's own back
    fun installDriver() {
        val p = driverFor ?: return
        if (driverBusy) return
        driverBusy = true
        driverNote = ""
        thread(name = "lizard-usb-driver", isDaemon = true) {
            val why = UsbDriver.install(p)
            post {
                driverBusy = false
                if (why.isEmpty()) { driverFor = null; pickUsbPhone = true; pickCamera = p.camera; refreshCameras() }
                else driverNote = "The USB driver was not installed: $why"
            }
        }
    }
    fun removeDriver() {
        if (driverBusy) return
        driverBusy = true
        driverNote = ""
        thread(name = "lizard-usb-driver", isDaemon = true) {
            val why = UsbDriver.remove()
            post {
                driverBusy = false
                if (why.isNotEmpty()) driverNote = "The USB driver was not removed: $why"
                refreshCameras()
            }
        }
    }

    // ai: The camera list kept current with no interaction (2026-10-27): a second at a time while Receive shows, the
    // ai: mark of what is plugged in (Session.camerasStamp: cheap, read off the UI thread) and the list read again
    // ai: (refreshCameras) only where it changed, a camera that came or went shown so
    private val watchTimer = Timer(1000) { watchCameras() }
    private var stamp: String? = null
    private var stamping = false
    // ai: a USB phone gone from the list (unplugged), the camera chosen and not reading (a stream it was in has ended by
    // ai: then): let go, the camera closed and the choice the first listed (the saved choice kept, for its return)
    private fun dropGonePhone() {
        val c = camera
        if (!c.startsWith("phone:usb:") || cameras.any { it.id == c } || running) return
        camera = ""
        info = null
        mode = null
        session?.closeCamera()
    }
    private fun watchCameras() {
        dropGonePhone()
        if (stamping || app.loadError.isNotEmpty()) return
        stamping = true
        thread(name = "lizard-camera-watch", isDaemon = true) {
            val st = runCatching { Session.camerasStamp() }.getOrNull()
            post {
                stamping = false
                if (st != null && st != stamp) {
                    val first = stamp == null
                    stamp = st
                    if (!first) refreshCameras()   // ai: the first mark only taken (the Camera menu lists as it opens)
                }
            }
        }
    }

    // ai: Home's lists: the received files (a small file, read here) and the unfinished transfers (a folder each); the
    // ai: takes not in Downloads filed again (retryKept); the cameras' settings moved once (moving)
    fun refreshLists() {
        if (app.loadError.isEmpty()) moving()
        files = Received.list()
        retryKept()
        thread(name = "lizard-partials", isDaemon = true) { val p = Partials.list(); post { partials = p } }
    }
    // ai: an unfinished transfer deleted (Home's Delete): forgotten by the session where one is (the receiver's, or its
    // ai: folder), else its folder deleted on the files' thread unless another window's session holds the data folder (a
    // ai: session of this window let go a moment ago and still closing is waited for: session.h deletePartial)
    fun deletePartial(p: Partial) {
        partials = partials.filter { it.id != p.id }
        val s = session
        if (s != null) s.forget(p.root.ifEmpty { p.id })
        else later {
            val why = Session.deletePartial(Native.data.path, p.id)
            if (why.isNotEmpty()) System.err.println("receive: ${p.id} not deleted: $why")
            val l = Partials.list(); post { partials = l }
        }
    }
    // ai: a received file deleted (Home's Delete, Delete all): to the trash, on the files' thread; the session lets go
    // ai: the transfer where the receiver holds that root (session.h clear), so the line's "Received" goes and the same
    // ai: file in the light is received and filed anew (the app's forgetReceived), the camera stopped for it again
    fun deleteReceived(e: Received.Entry) = deleteAll(listOf(e))
    fun deleteAll(es: List<Received.Entry> = files) {
        val gone = es.map { it.path }.toSet()
        files = files.filter { it.path !in gone }
        val roots = es.map { it.root }.filter { it.isNotEmpty() }.toSet()
        if (root.isNotEmpty() && root in roots) session?.clear()
        if (noteRoot in roots) { noteRoot = ""; fileNote = "" }
        later { es.forEach { runCatching { Received.delete(it) } }; val l = Received.list(); post { files = l } }
    }

    // ai: the takes not in Downloads filed (Received.retry), one pass queued at a time: at each look at the lists (Home,
    // ai: Receive entered; the first is the launch's, which files any a close cut short), at each Start, and each time
    // ai: the session takes a file (its stats' kept.taken)
    private var retrying = false
    private var again = false
    private fun retryKept() {
        if (retrying) { again = true; return }
        retrying = true
        later {
            val done = Received.retry()
            val l = Received.list()
            post {
                retrying = false; files = l; noted(done)
                if (again) { again = false; retryKept() }
            }
        }
    }
    // ai: What filings came to (the UI thread): the note set where the file the receiver holds is not in Downloads, and
    // ai: cleared once the file it names is
    private var noteRoot = ""
    private fun noted(fs: List<Received.Filed>) {
        for (f in fs) {
            if (f.why.isEmpty()) { if (f.root == noteRoot) { noteRoot = ""; fileNote = "" } }
            else if (f.root == root || f.root == takenRoot) { noteRoot = f.root; fileNote = "Not saved to Downloads: ${f.why}" }
        }
    }

    // ai: The stats, every quarter second while the session lives (its stats: the receiver's, the session's beside them)
    private val timer = Timer(250) { poll() }
    private var tick = 0
    private var taken = -1L          // ai: the session's takes as last counted (-1: none read yet)
    private var takenRoot = ""
    private var camJson = ""         // ai: the session's camera() as last read
    private var keptFor = ""         // ai: "<id>/<layout>" the zoom, exposure and gain shown were taken up for
    private fun poll() {
        val s = session ?: return
        val raw0 = runCatching { s.stats }.getOrDefault("")
        if (raw0.isEmpty()) return
        raw = raw0
        val j = parse(raw0)
        stats = j
        val r = rxOf(j)
        rx = r
        val f = j?.get("file") as? JsonObject
        root = f?.str("root").orEmpty()
        val ph = rxPhaseOf(j)
        if (ph != phase) phase = ph
        if (phase is RxPhase.Error) pendingMode = null   // ai: a chosen mode that would not start: the field shows what is
        if (decoder == "auto" && running) j?.str("autoRan")?.takeIf { it.isNotEmpty() }?.let { autoRan = it }
        // ai: a phone's or an upload's own crop is the receiver's (the session makes it so at the start)
        if (fixed) j?.str("layout")?.takeIf { it == "1:1" || it == "2:1" }?.let { if (it != layout) layout = it }
        cameraTick(s)
        // ai: a file the session took into kept/: filed into Downloads; a take that failed said under the state line
        val k = j?.get("kept") as? JsonObject
        val n = k?.num("taken")?.toLong() ?: 0L
        takenRoot = k?.str("root").orEmpty()
        if (n != taken) { val first = taken < 0; taken = n; if (!first || n > 0) retryKept() }
        k?.str("error")?.takeIf { it.isNotEmpty() && k.str("root") == root }?.let { if (noteRoot != root) { noteRoot = root; fileNote = "Not saved: $it" } }
        runsTick(j)
        letGo()
        if (tick++ % 4 == 0 && app.screen == Screen.Home) thread(name = "lizard-partials", isDaemon = true) { val p = Partials.list(); post { partials = p } }
    }
    // ai: the camera open as the session describes it: Settings' camera, its modes and the mode running or next; its
    // ai: zoom, exposure and gain as kept for it, taken up when it opens or its crop changes (not while a bar moves)
    private fun cameraTick(s: Session) {
        val cj = runCatching { s.camera }.getOrDefault("")
        // ai: the camera chosen not open after a refusal (an open refused): none shown as open; one being opened keeps the
        // ai: one before shown until it is
        if (cj == "{}" && phase is RxPhase.Error && info?.id != camera) { info = null; mode = null }
        if (cj == camJson) return
        camJson = cj
        val j = parse(cj)
        if (pendingMode != null && j?.let { camInfo(it) } == null) pendingStopped = true
        val ci = j?.let { camInfo(it) } ?: return
        info = ci
        val started = j.num("mode").toInt()
        val kept = j["kept"] as? JsonObject
        // ai: a mode just chosen (pickMode) stays the field's until the camera reports a stream started after its stop: in
        // ai: between, the session describes the reopened camera with no stream and its kept mode still the old one, which
        // ai: put the old mode back in the menu for a moment (2026-11-02; on a slow camera's reopen every time)
        val pm = pendingMode
        if (pm != null) {
            if (started < 0) pendingStopped = true
            else if (pendingStopped) { pendingMode = null; mode = ci.modes.getOrNull(started) ?: mode }
        } else mode = ci.modes.getOrNull(if (started >= 0) started else kept?.num("mode")?.toInt() ?: -1) ?: mode
        val key = "${ci.id}/$layout"
        if (key != keptFor) {
            keptFor = key
            zoom = kept?.numOrNull("zoom") ?: ci.control("zoom")?.value
            exposure = kept?.numOrNull("exposure")
            gain = kept?.numOrNull("gain")
        }
    }
    // ai: the session's replays (its stats' "replays"), as Developer Tools lists them
    private fun runsTick(j: JsonObject?) {
        val rs = runsOf(j) ?: return
        if (rs != runs.toList()) { runs.clear(); runs.addAll(rs) }
    }

    // ai: The session let go off Receive (2026-10-27; a window that had received once and then sent held the lock and
    // ai: refused the other's Start): a window off Receive with its camera stopped and no replay recording or saving lets
    // ai: its session go on a thread of its own (session.h's destructor: the camera closed, the receiver let go, the
    // ai: lock released), so a window that sends holds none. The transfer in hand stays in the store, which the next
    // ai: session (the next Receive, here or in another window) resumes when its header is read again; a session made
    // ai: here meanwhile waits for the lock on its own thread.
    private fun letGo() {
        val s = session ?: return
        if (app.screen == Screen.Receive || running) return
        if (runs.any { it.state == RunState.Recording || it.state == RunState.Ending }) return
        session = null
        seq = 0L; preview = null
        timer.stop()
        closing = thread(name = "lizard-rx-close", isDaemon = true) { runCatching { s.close() } }
    }
    private var closing: Thread? = null   // ai: a session let go and still closing (the app's close waits for it)

    // ai: The preview: the newest frame's crop as the decoder reads it, at most PREVIEW px on its long side, grey, into
    // ai: one of two Skia bitmaps in turn (installPixels copies the bytes; Compose draws a bitmap by its copy), while On,
    // ai: pulled at each frame the window draws (DesktopReceive.Picture: Compose's frame clock; a 16 ms Swing timer
    // ai: beside the display's refresh until 2026-11-02, its frames now and then held two refreshes or skipped)
    private val buf = ByteArray(PREVIEW * PREVIEW)
    private var seq = 0L
    private val bitmaps = arrayOfNulls<Bitmap>(2)
    private var flip = 0
    fun pull() {
        val s = session ?: return
        if (phase != RxPhase.On) return
        val got = runCatching { s.preview(seq, PREVIEW, buf) }.getOrNull() ?: return
        if (got.size < 3) return
        val w = got[1].toInt(); val ht = got[2].toInt()
        if (w <= 0 || ht <= 0 || w.toLong() * ht > buf.size) return
        seq = got[0]
        flip = flip xor 1
        val bm = bitmaps[flip] ?: Bitmap().also { bitmaps[flip] = it }
        if (bm.installPixels(ImageInfo(w, ht, ColorType.GRAY_8, ColorAlphaType.OPAQUE), buf, w)) preview = bm.asComposeImageBitmap()
    }

    // ai: What Receive shows
    val line: Readout.Line get() = Readout.line(on, phase == RxPhase.Starting, phase == RxPhase.Loading, (phase as? RxPhase.Error)?.why, rx,
        files.firstOrNull { it.root == root && it.bundle }?.files ?: 0)
    // ai: the lab line while the figures are this stream's (live), none otherwise (Readout.lab's while the camera is off)
    val lab: String get() = Readout.lab(live, rx)
    // ai: the preview's box: the crop's shape (1:1; 2:1 along the camera's long side)
    val cropRatio: Float get() = if (layout != "2:1") 1f else if ((mode?.h ?: 0) > (mode?.w ?: 1)) 0.5f else 2f
    // ai: One hint, the first that holds, while the camera runs and a word is read (dev.lizard.ui hintOf): no camera of a
    // ai: PC's is held by the phase lock; the mode's rate where the camera agreed to none
    // ai: locked: the phase lock on the camera (a GenICam camera's software trigger, cam/aravis.cpp, since 2026-11-01;
    // ai: UVC's lock is {"off": why}, a stream whose trigger did not arm {})
    val hint: String get() = hintOf(stats, on, phone, upload,
        locked = (stats?.get("lock") as? JsonObject)?.let { it.isNotEmpty() && "off" !in it } == true, modeFps = mode?.fps ?: 0.0)

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
            val named = if (v > 0) "LIZARD-${Pick.STEP * v}${word?.num("fps")?.toInt()?.takeIf { it > 0 }?.let { " at $it fps" } ?: ""}" else "no word yet"
            val told = (j["totals"] as? JsonObject)?.let { t -> "total ${t.num("blocks").toLong()} blocks in ${t.num("frames").toLong()} frames" }
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
        // ai: the most the close waits for the files' thread (a large tar's unpack); what is left is done at the next launch
        const val CLOSE_FILING_SECS = 30L
        const val MOVE_WAIT_MS = 15_000L   // ai: the most a first session waits for the cameras' settings to be moved

        private fun f(x: Double, d: Int) = String.format(Locale.ROOT, "%.${d}f", x)
        // ai: a rate as a menu shows it: whole where it is (30, 60), else two decimals (29.97)
        fun fpsText(v: Double) = if (abs(v - v.roundToInt()) < 0.01) "${v.roundToInt()}" else String.format(Locale.ROOT, "%.2f", v)

    }
}

// ai: a number, or null where the field is none or null (the session's kept settings)
private fun JsonObject.numOrNull(k: String) = (this[k] as? JsonPrimitive)?.doubleOrNull
