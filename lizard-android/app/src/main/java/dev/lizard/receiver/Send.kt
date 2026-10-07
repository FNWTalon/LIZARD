package dev.lizard.receiver

import android.content.Context
import android.content.res.Configuration
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.movableContentOf
import androidx.compose.runtime.remember
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import android.util.Log
import android.view.Choreographer
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import androidx.annotation.RequiresApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.floor
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.roundToInt
import java.util.Locale
import androidx.activity.compose.BackHandler
import androidx.compose.runtime.DisposableEffect
import androidx.compose.foundation.interaction.MutableInteractionSource

// ai: Send (2026-10-01): the web sender's job on the phone, the C on the CPU painting (liblizard/core/tx/sender.h). A
// ai: file chosen here or shared to Lizard from another app is copied into the cache and mapped by the sender; the
// ai: format is the largest the code's square holds (Pick, the web's pickVersion), in the default ring, at Advanced's
// ai: rate (60 unless set; until 2026-10-02 "Receiving with" named it, the app 60, a browser 24); frames go onto a
// ai: SurfaceView at that pace (a Choreographer callback each vsync posting the next painted frame when one is due; the
// ai: surface asks the display for that rate, the switch always made), the screen kept on, at full brightness, its bars
// ai: hidden. The test stream is Advanced's. Vsync-locked since 2026-10-02: on
// ai: Android 13 and up each frame is due by its frame timeline's expected presentation time and posted for that vsync
// ai: (jni.cpp Ring), not into the window's queue at whatever refresh it is latched for; `adb shell setprop
// ai: debug.lizard.sendvsync 0` before Start takes the window's queue, for an A/B.
class SendState(private val a: MainActivity) {
    sealed interface Phase { data object Idle : Phase; data object Preparing : Phase; data object On : Phase; data class Error(val why: String) : Phase }
    data class Stats(val label: String = "", val shownFps: Double = 0.0, val offeredKBs: Double = 0.0, val paintMs: Double = 0.0, val pass: Double = 0.0,
                     val painter: String = "", val gpuWhy: String = "", val sentBytes: Long = 0,   // ai: sentBytes: the file's bytes as they go (2026-10-05)
                     // ai: the GPU ring's tally over the last second (2026-10-07, jni.cpp txPostStats): posted, held by
                     // ai: vsyncs, late, behind, buffers filled of the ring's
                     val gpuRing: Boolean = false, val posted: Long = 0, val held: String = "", val late: Long = 0, val behind: Long = 0, val filled: Int = 0, val slots: Int = 0)

    private val prefs = a.getSharedPreferences("lizard", Context.MODE_PRIVATE)
    var uri by mutableStateOf<Uri?>(null)
    var name by mutableStateOf("")
    var size by mutableStateOf(0L)
    var type by mutableStateOf("")
    var test by mutableStateOf(false)
    // ai: Blocks a frame (2026-10-02), the web sender's
    // ai: slider (send.html #blocks and #subchAuto): 0 for auto, the largest the code's square holds (Pick.pick), else 1
    // ai: to 128 set by hand, a block 8 sub-channels, painted as set whatever the square (the web's too); kept, as the
    // ai: web's #subch is. picked: the sub-channels the last configure took, which auto's label names.
    var blocks by mutableStateOf(prefs.getInt("sendBlocks", 0))
    var picked by mutableStateOf(0)
    // ai: Pictures a second (2026-10-02), the web's #fps: 1 to 60, 60 unless set (the word states it; a rate the
    // ai: display's refresh does not divide holds pictures unevenly); kept, as the web's is. "Receiving with" (the app
    // ai: 60, a browser 24) went the same day.
    var rate by mutableStateOf(prefs.getInt("sendFps", 60).let { if (it in 1..60) it else 60 })
    // ai: The screen's brightness while sending (2026-10-02), 1 to 100%, 100 at every launch (2026-10-03; kept as
    // ai: `sendBrightness` before, so a slider moved once dimmed
    // ai: every later send). The window's own override, so the phone's setting is left as it
    // ai: was once sending stops; set at Start (screenFor), the settings being the idle screen's.
    var brightness by mutableStateOf(100)
    // ai: Painting with (2026-10-02; the web's encoder switch): auto (the GPU
    // ai: where it builds and its first frames are the C's, else the CPU), the GPU (liblizard/core/tx/gpu_painter.h) or the CPU's
    // ai: painters; kept.
    var painter by mutableStateOf(prefs.getString("sendPainter", "auto") ?: "auto")
    // ai: Codes (2026-10-05, the web's #codes and #gap): one, or two side by side for a receiver's 2:1 crop, `gap` modules
    // ai: apart (0 to 64, 12 unless set); kept. The format is picked for one code's share of the box (roomOf).
    var codes by mutableStateOf(prefs.getInt("sendCodes", 1).coerceIn(1, 2))
    var gap by mutableStateOf(prefs.getInt("sendGap", 12).coerceIn(0, 64))
    // ai: the code box's width over its height: the codes and the gaps between them, each code as tall as the box
    val aspect get() = codes + (codes - 1) * Pick.gapShare(gap)
    var phase by mutableStateOf<Phase>(Phase.Idle)
    // ai: Paused (2026-10-02): nothing presented, the run
    // ai: kept, the last picture left on screen (a receiver holds what it has); a resume presents at once (tick)
    var paused by mutableStateOf(false)
    var stats by mutableStateOf(Stats())

    private var h = 0L
    private var surface: Surface? = null
    var shown by mutableStateOf(false)   // ai: a picture has been presented since the start (Send.kt CodeBox's overlay)
    private var room = 0
    private var configured = 0          // ai: the room the sender was last configured for
    private var due = 0L
    private var lastStats = 0L
    private var vsynced = false         // ai: this send's frames posted for their vsync (Android 13 on; Ring in jni.cpp)
    private var assets = ""             // ai: the generated tree (Assets.ensure), the GPU painter's kernels and tables
    private var vw = 0                  // ai: the view's size in pixels, the vsync-locked present's destination
    private var vh = 0
    val fps get() = rate

    fun pick(u: Uri) {
        uri = u; test = false
        name = "a file"; size = 0; type = a.contentResolver.getType(u) ?: ""
        runCatching {
            a.contentResolver.query(u, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
                if (c.moveToFirst()) { name = c.getString(0) ?: name; size = if (c.isNull(1)) 0 else c.getLong(1) }
            }
        }
        if (phase is Phase.Error) phase = Phase.Idle
    }
    fun chooseBlocks(b: Int) { blocks = b.coerceIn(0, 128); prefs.edit().putInt("sendBlocks", blocks).apply(); if (phase == Phase.On) reconfigure() }
    fun chooseRate(r: Int) { rate = r.coerceIn(1, 60); prefs.edit().putInt("sendFps", rate).apply(); if (phase == Phase.On) reconfigure() }
    fun chooseBrightness(b: Int) { brightness = b.coerceIn(1, 100) }
    fun pause(p: Boolean) { if (phase == Phase.On) paused = p }
    // ai: what the code offers at most, KB/s: blocks a frame x 469 B x pictures a second; null before the first configure
    val capacityKBs get() = if (picked > 0) codes * (picked / 8) * 469.0 * rate / 1000 else null
    fun choosePainter(p: String) { painter = p; prefs.edit().putString("sendPainter", p).apply(); if (phase == Phase.On) reconfigure() }
    fun chooseCodes(c: Int) { codes = c.coerceIn(1, 2); prefs.edit().putInt("sendCodes", codes).apply(); if (phase == Phase.On) reconfigure() }
    fun chooseGap(g: Int) { gap = g.coerceIn(0, 64); prefs.edit().putInt("sendGap", gap).apply(); if (phase == Phase.On) reconfigure() }
    // ai: one code's room in the view, px: the box's height, or its width's share where that is less
    private fun roomOf() = min((vw / aspect).toInt(), vh)
    // ai: the column's edge let go (2026-10-05): the format re-picked for the box it left
    fun resized() { if (phase == Phase.On && roomOf() != configured) reconfigure() }

    // ai: the file into the cache (a document is a stream; the sender maps a file), then the sender and its format
    fun start() {
        if (phase == Phase.Preparing || phase == Phase.On) return
        val u = uri
        if (!test && u == null) return
        phase = Phase.Preparing
        shown = false
        thread(name = "lizard-send-prep") {
            val dir = File(a.cacheDir, "send").apply { deleteRecursively(); mkdirs() }
            val path = if (test) "" else try {
                val f = File(dir, "file")
                a.contentResolver.openInputStream(u!!)!!.use { i -> f.outputStream().use { o -> i.copyTo(o, 1 shl 20) } }
                f.path
            } catch (e: Exception) { a.runOnUiThread { phase = Phase.Error("The file could not be read: ${e.message}") }; return@thread }
            val hnd = Native.txCreate(path, if (test) "" else name, if (test) "" else type)
            val err = if (hnd == 0L) Native.txError() else ""
            // ai: the GPU painter's device and pipelines made here, off the main thread that configures
            if (hnd != 0L && painter != "cpu") {
                assets = Assets.ensure(a).path
                val why = Native.txPrepare(hnd, assets)
                if (why.isNotEmpty()) Log.i(Engine.TAG, "send: no GPU painter: $why")
            }
            a.runOnUiThread {
                if (hnd == 0L) { phase = Phase.Error("This file cannot be sent: $err"); return@runOnUiThread }
                if (phase != Phase.Preparing) { Native.txDestroy(hnd); return@runOnUiThread }   // ai: stopped meanwhile
                h = hnd; configured = 0; phase = Phase.On
                vsynced = Build.VERSION.SDK_INT >= 33 && Native.prop("debug.lizard.sendvsync") != "0"
                screenFor(true)
                reconfigure()
                due = 0; lastStats = 0; lastTimeline = 0; vsyncNs = 0
                Log.i(Engine.TAG, "send: ${if (vsynced) "frames posted for their vsync" else "frames into the window's queue"}")
                if (vsynced && Build.VERSION.SDK_INT >= 33) Choreographer.getInstance().postVsyncCallback(vsync)
                else Choreographer.getInstance().postFrameCallback(frame)
            }
        }
    }

    fun stop() {
        Choreographer.getInstance().removeFrameCallback(frame)
        if (Build.VERSION.SDK_INT >= 33) Choreographer.getInstance().removeVsyncCallback(vsync)
        if (h != 0L) { Native.txDestroy(h); h = 0 }
        if (phase == Phase.On || phase == Phase.Preparing) phase = Phase.Idle
        shown = false
        stats = Stats(); paused = false
        screenFor(false)
    }

    // ai: the format the code's square holds now (a turn of the phone re-picks it; the transfer goes on)
    private fun reconfigure() {
        room = roomOf()
        if (h == 0L || room <= 0) return
        val subch = if (blocks > 0) 8 * blocks else Pick.pick(room.toDouble())
        val threads = min(4, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val err = Native.txConfigure(h, Pick.nFor(subch), subch, Pick.span(), fps, threads, when (painter) { "cpu" -> 0; "gpu" -> 1; else -> 2 }, assets, codes, gap, vsynced)
        if (err.isNotEmpty()) { phase = Phase.Error(err); stop(); return }
        configured = room
        picked = subch
        // ai: the window's queue's vote (the vsync-locked present votes on its own surface, jni.cpp): the switch always
        // ai: made from Android 12 (the default makes it only where seamless, which can leave the panel at another rate)
        runCatching {
            if (Build.VERSION.SDK_INT >= 31) surface?.setFrameRate(fps.toFloat(), Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE, Surface.CHANGE_FRAME_RATE_ALWAYS)
            else if (Build.VERSION.SDK_INT >= 30) surface?.setFrameRate(fps.toFloat(), Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE)
        }
        val st = runCatching { JSONObject(Native.txStats(h)) }.getOrNull()
        Log.i(Engine.TAG, "send: LIZARD-$subch${if (codes > 1) " x $codes, $gap modules apart" else ""}${if (blocks > 0) " set by hand" else ""} in a $room px square at $fps a second, painting on the " +
            (if (st?.optString("painter") == "gpu") "GPU (${st.optString("device")})" else "CPU, $threads painters") +
            (st?.optString("gpuWhy").orEmpty().let { if (it.isNotEmpty()) ", not the GPU: $it" else "" }))
    }

    internal val holder = object : SurfaceHolder.Callback {
        override fun surfaceCreated(hd: SurfaceHolder) { surface = hd.surface }
        override fun surfaceChanged(hd: SurfaceHolder, format: Int, w: Int, hh: Int) {
            surface = hd.surface; vw = w; vh = hh
            // ai: not while the column's edge is being dragged: the code's box changes every frame of it (resized after)
            if (phase == Phase.On && roomOf() != configured && !a.sideDragging) reconfigure()
        }
        override fun surfaceDestroyed(hd: SurfaceHolder) { surface = null }
    }

    // ai: each vsync: the next painted frame when one is due at the asked rate (half a vsync early is on time). t: when the
    // ai: frame will be shown (the frame timeline's expected presentation time) where vsync names that vsync, else the
    // ai: callback's frame time, the window's queue showing it a refresh or two after.
    private fun tick(t: Long, vsync: Long) {
        val period = 1_000_000_000L / fps
        val s = surface
        if (paused) due = 0L   // ai: presented at once on a resume
        else if (s != null && configured > 0 && t >= due - 4_000_000L) {
            if (Native.txPresent(h, s, vsync, vw, vh)) { due = if (due == 0L || t - due > period) t + period else due + period; if (!shown) shown = true }
        }
        pollStats(t)
    }
    // ai: The vsync path's schedule (2026-10-07: "present on an exact fixed-refresh schedule"): the display's period
    // ai: from the frame timelines' spacing; a picture every round(period / asked) vsyncs where that is whole within
    // ai: 1%, else on the asked rate's grid (the web's due rule); for every timeline the Choreographer offers (the
    // ai: preferred one and the ones after it) not yet passed, the next filled buffer is posted for it where a picture
    // ai: is due there (Native.txPost), so the compositor holds the pictures ahead. Nothing filled at a due vsync: the
    // ai: picture on screen stays (counted behind in the ring's tally) and the next picture takes the next due vsync.
    private var vsyncNs = 0L
    private var lastTimeline = 0L
    @RequiresApi(33)
    private fun tickVsync(d: Choreographer.FrameData) {
        val tls = d.frameTimelines.sortedBy { it.expectedPresentationTimeNanos }
        if (tls.size >= 2) { val p = tls[1].expectedPresentationTimeNanos - tls[0].expectedPresentationTimeNanos; if (p > 1_000_000L) vsyncNs = p }
        val s = surface
        val askedNs = 1_000_000_000L / fps
        val perV = if (vsyncNs > 0) askedNs.toDouble() / vsyncNs else 0.0
        val per = Math.round(perV)
        val periodNs = if (per >= 1 && vsyncNs > 0 && abs(perV - per) <= 0.01 * per) per * vsyncNs else askedNs
        if (paused) { due = 0L; lastTimeline = tls.lastOrNull()?.expectedPresentationTimeNanos ?: lastTimeline }
        else for (tl in tls) {
            val t = tl.expectedPresentationTimeNanos
            if (t <= lastTimeline) continue
            lastTimeline = t
            if (s == null || configured <= 0 || t + vsyncNs / 2 < due) continue
            if (Native.txPost(h, s, tl.vsyncId, t, vsyncNs, vw, vh)) { due = if (due == 0L || t - due > periodNs) t + periodNs else due + periodNs; if (!shown) shown = true }
            else due = t + periodNs
        }
        pollStats(tls.firstOrNull()?.expectedPresentationTimeNanos ?: System.nanoTime())
    }
    private fun pollStats(t: Long) {
        if (t - lastStats <= 1_000_000_000L) return
        lastStats = t
        runCatching {
            val j = JSONObject(Native.txStats(h))
            val p = if (vsynced) JSONObject(Native.txPostStats(h)) else null
            val held = p?.optJSONObject("held")?.let { o -> o.keys().asSequence().sortedBy { it.toIntOrNull() ?: 0 }.joinToString(", ") { "$it: ${o.opt(it)}" } }.orEmpty()
            stats = Stats(j.optString("label"), j.optDouble("shownFps"), j.optDouble("offeredKBs"), j.optDouble("paintMs"), j.optDouble("pass", 0.0),
                j.optString("painter"), j.optString("gpuWhy"), j.optLong("sentBytes", 0),
                p?.optBoolean("gpu") ?: false, p?.optLong("posted") ?: 0, held, p?.optLong("late") ?: 0, p?.optLong("behind") ?: 0, p?.optInt("filled") ?: 0, p?.optInt("slots") ?: 0)
            if (p?.optBoolean("gpu") == true) Log.i(Engine.TAG, "send: posted ${stats.posted}, held {${held}}, late ${stats.late}, behind ${stats.behind}, ${stats.filled} of ${stats.slots} buffers filled, ${j.optInt("ahead")} of ${j.optInt("depth")} frames painted ahead")
            if (j.optString("error").isNotEmpty()) phase = Phase.Error(j.optString("error"))
            if (p != null && p.optString("error").isNotEmpty()) phase = Phase.Error(p.optString("error"))
        }
    }
    private val frame = object : Choreographer.FrameCallback {
        override fun doFrame(t: Long) {
            if (h == 0L) return
            tick(t, 0)
            Choreographer.getInstance().postFrameCallback(this)
        }
    }
    // ai: Android 13 on: the preferred frame timeline, the one the system expects a frame started now to make
    private val vsync: Choreographer.VsyncCallback by lazy @RequiresApi(33) {
        object : Choreographer.VsyncCallback {
            override fun onVsync(d: Choreographer.FrameData) {
                if (h == 0L) return
                tickVsync(d)
                Choreographer.getInstance().postVsyncCallback(this)
            }
        }
    }

    // ai: while sending: the screen on, at full brightness, its bars hidden; all put back after
    private fun screenFor(on: Boolean) {
        val w = a.window
        val c = WindowCompat.getInsetsController(w, w.decorView)
        if (on) {
            w.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            w.attributes = w.attributes.apply { screenBrightness = brightness / 100f }
            c.hide(WindowInsetsCompat.Type.systemBars())
        } else {
            w.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            w.attributes = w.attributes.apply { screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE }
            c.show(WindowInsetsCompat.Type.systemBars())
        }
    }
}

// ai: Send, laid out as Receive since 2026-10-05 (a setup page, then a separate sending screen, until then): the code's
// ai: box where Receive has the camera, the code painted in it while sending and, idle, the file to send (a tap chooses
// ai: one, as the web sender's page); under it the state, the figures, the buttons and the rate, then Settings,
// ai: Developer Tools and About. Landscape: the column on the left and the code on the right, as large as the rest allows;
// ai: the column folds to a rail (play, pause or resume, the code's capacity, Home at its foot), as Receive's.
@Composable
internal fun MainActivity.SendScreen() {
    val s = send
    val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    // ai: one code view for both layouts: a turn of the phone moves the surface rather than making another
    val code = remember { movableContentOf<Modifier> { m -> CodeBox(m) } }
    // ai: Fullscreen (2026-10-06, the web's #fs): the code alone on the page's white, as large as the screen allows,
    // ai: the system bars hidden; a tap on it or Back brings the screen back (the bars too, unless a send keeps them
    // ai: hidden). The code's surface moves, not remade (movableContentOf).
    var full by remember { mutableStateOf(false) }
    if (full) {
        BackHandler { full = false }
        DisposableEffect(Unit) {
            val c = WindowCompat.getInsetsController(window, window.decorView)
            c.hide(WindowInsetsCompat.Type.systemBars())
            onDispose { if (send.phase != SendState.Phase.On && send.phase != SendState.Phase.Preparing) c.show(WindowInsetsCompat.Type.systemBars()) }
        }
        BoxWithConstraints(Modifier.fillMaxSize().background(Bg).clickable(indication = null, interactionSource = remember { MutableInteractionSource() }) { full = false },
            contentAlignment = Alignment.Center) {
            val a = s.aspect.toFloat()
            val w = minOf(maxWidth, maxHeight * a)
            code(Modifier.size(w, w / a))
        }
        return
    }
    val side: @Composable () -> Unit = {
        SendPanel { full = true }
        Fold("Settings", isOpen("sendSettings"), { toggle("sendSettings") }) {
            Fields {
                // ai: the fields in the web's shape, a menu each (2026-10-06; chips until then); with nothing chosen the setting
                // ai: stays auto and the menu shows the painter running; a choice takes that painter outright
                Select("Encoder", if (s.painter == "auto") s.stats.painter.ifEmpty { "gpu" } else s.painter, listOf("gpu" to "GPU", "cpu" to "CPU")) { s.choosePainter(it) }
                Field("Brightness", "${s.brightness}%") {
                    Bar(s.brightness.toFloat(), 1f..100f, 98) { v -> val n = v.roundToInt(); if (n != s.brightness) s.chooseBrightness(n) }
                }
                Group("Code") {
                    AutoSlider("Blocks", s.blocks, s.picked / 8, 1..128, { n -> "$n, ${"%.1f".format(Locale.ROOT, n * 469 / 1000.0)}\u00a0KB" }) { s.chooseBlocks(it) }
                    RateSlider(s)
                    // ai: one code, or two side by side for a receiver's 2:1 crop, and their gap (the web's #codes and #gap)
                    Select("Codes", s.codes, listOf(1 to "One", 2 to "Two")) { s.chooseCodes(it) }
                    if (s.codes == 2) Field("Gap", "${s.gap} modules") { Bar(s.gap.toFloat(), 0f..64f, 63) { v -> val g = v.roundToInt(); if (g != s.gap) s.chooseGap(g) } }
                }
            }
        }
        Fold("Developer Tools", isOpen("sendAdvanced"), { toggle("sendAdvanced") }) {
            Fields {
                Select("Payload", s.test, listOf(false to "File", true to "Test stream"), enabled = s.phase != SendState.Phase.On && s.phase != SendState.Phase.Preparing) { s.test = it }
            }
            sendLab(s).let { if (it.isNotEmpty()) CodeBlock(it) }
        }
        About()
        Spacer(Modifier.height(24.dp))
    }
    if (!landscape) Column(Modifier.fillMaxSize().background(Bg).safeDrawingPadding().padding(horizontal = 16.dp)) {
        TopBar("Send", onBack = { go(MainActivity.Screen.Home) }) { SwapBtn("Receiver") { go(MainActivity.Screen.Receive) } }
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            Spacer(Modifier.height(8.dp))
            code(Modifier.fillMaxWidth().aspectRatio(s.aspect.toFloat()))
            side()
        }
    } else BoxWithConstraints(Modifier.fillMaxSize().background(Bg).safeDrawingPadding()) {
        // ai: no narrower than Start, Stop or Resume beside Fullscreen with no label folded (2026-10-06)
        val sideLo = CellsFloor(listOf("Start", "Stop", "Resume", "Fullscreen"), 2)
        val sideW = SideWidth(maxWidth, sideDp, sideLo)
        val screenW = maxWidth
        Row(Modifier.fillMaxSize()) {
            if (isOpen("sendCollapsed")) Rail(onClick = { toggle("sendCollapsed") },
                foot = { Square { IconBtn(R.drawable.ic_back, "Home") { go(MainActivity.Screen.Home) } } }) { SendRail() }
            else Column(Modifier.width(sideW).fillMaxHeight().padding(horizontal = 16.dp)) {
                TopBar("Send", onBack = null) { CollapseBtn(false) { toggle("sendCollapsed") } }
                Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) { side() }
            }
            // ai: the column's edge, dragged to resize it (2026-10-05); the code re-picked once the finger is off it
            if (isOpen("sendCollapsed")) VerticalDivider(color = Line)
            else SideEdge(sideW, screenW, sideLo, onStart = { sideDragging = true }, onDrag = { sideDp = it.value }, onDone = { sideDragging = false; saveSide(); s.resized() })
            BoxWithConstraints(Modifier.weight(1f).fillMaxHeight().padding(16.dp), contentAlignment = Alignment.Center) {
                val a = s.aspect.toFloat()
                val w = minOf(maxWidth, maxHeight * a)
                code(Modifier.size(w, w / a))   // ai: square for one code, about 2:1 for two
            }
        }
    }
}

// ai: The code's box: while sending the code itself, painted into a SurfaceView the box's size (SendState.holder); idle,
// ai: on the soft fill as Receive's camera-off cover, the file to send or "Choose a file", a tap choosing one.
@Composable
private fun MainActivity.CodeBox(modifier: Modifier) {
    val s = send
    val sending = s.phase == SendState.Phase.On || s.phase == SendState.Phase.Preparing
    // ai: the file's look stays over the surface until the first picture is presented (2026-10-07): a SurfaceView is
    // ai: black from its making until its first frame, through Preparing (a file compressed) and the first presents,
    // ai: and a box drawn over it in the window hides it
    val shown = sending && s.shown
    Box(modifier.clipToBounds().background(if (shown) Bg else Soft), contentAlignment = Alignment.Center) {
        if (sending) AndroidView({ ctx -> SurfaceView(ctx).apply { holder.addCallback(s.holder) } }, Modifier.fillMaxSize())
        if (!shown) Column(Modifier.fillMaxSize().background(Soft).clickable(enabled = !s.test && !sending) { pickFile() }, verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(painterResource(R.drawable.ic_file), contentDescription = null, Modifier.size(24.dp), tint = Muted)
            Text(if (s.test) "Test stream" else if (s.uri == null) "Choose a file" else s.name, style = MaterialTheme.typography.bodyMedium, color = Muted,
                textAlign = TextAlign.Center, modifier = Modifier.padding(horizontal = 16.dp))
        }
    }
}

// ai: The state, the buttons and the rate, as Receive's transfer panel: the state line (the file and its size,
// ai: "Preparing", "Sending" with the bytes that go, "Paused", or what went wrong), then Start, or Stop (Resume beside it
// ai: while paused), then the rate the code carries.
@Composable
private fun MainActivity.SendPanel(onFull: () -> Unit) {
    val s = send
    val on = s.phase == SendState.Phase.On
    val busy = on || s.phase == SendState.Phase.Preparing
    val st = s.stats
    val what = if (s.test) "the test stream" else s.name
    // ai: the bytes that go, what compression left of the file once the sender says (its own size before); the file's
    // ai: own size and the pass are Developer Tools' (sendLab, 2026-10-05)
    val sized = if (s.test || s.size <= 0) what else "$what, ${Readout.bytes(if (st.sentBytes > 0) st.sentBytes else s.size)}"
    val err = (s.phase as? SendState.Phase.Error)?.why
    val line = when {
        err != null -> err
        s.phase == SendState.Phase.Preparing -> "Preparing $what"
        on && s.paused -> "Paused"
        on -> "Sending $sized"
        s.test -> "The test stream"
        s.uri != null -> sized
        else -> ""
    }
    // ai: the web sender's order (2026-10-06): the buttons first (Start or Stop beside Fullscreen; paused, Resume and
    // ai: Stop on one row and Fullscreen under them, so no label folds), then the state line, then the rate as the
    // ai: web's figures line, 14 sp grey
    Column(Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 4.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val fs: @Composable (Modifier) -> Unit = { m -> Btn("Fullscreen", modifier = m, onClick = onFull) }
        if (!busy) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Btn("Start", Kind.Primary, enabled = s.test || s.uri != null, modifier = Modifier.weight(1f)) { s.start() }
            fs(Modifier.weight(1f))
        } else if (s.paused) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Btn("Resume", Kind.Primary, modifier = Modifier.weight(1f)) { s.pause(false) }
                Btn("Stop", modifier = Modifier.weight(1f)) { s.stop() }
            }
            fs(Modifier.fillMaxWidth())
        } else Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Btn("Stop", modifier = Modifier.weight(1f)) { s.stop() }
            fs(Modifier.weight(1f))
        }
    }
    if (line.isNotEmpty()) Text(line, style = MaterialTheme.typography.titleMedium, color = if (err != null) Bad else Fg, modifier = Modifier.padding(top = 4.dp, bottom = 2.dp))
    if (on && st.offeredKBs > 0) Text(Readout.rate(st.offeredKBs), style = MaterialTheme.typography.bodyMedium, color = Muted, modifier = Modifier.padding(top = 2.dp))
    HeatWarning(heat, clocks, modifier = Modifier.padding(top = 8.dp))
    Spacer(Modifier.height(8.dp))
}

// ai: The rail's squares: play to start (or to resume), pause while sending, and the code's capacity while it is shown
@Composable
private fun MainActivity.SendRail() {
    val s = send
    val on = s.phase == SendState.Phase.On
    Square {
        IconBtn(if (on && !s.paused) R.drawable.ic_pause else R.drawable.ic_play, if (!on) "Start" else if (s.paused) "Resume" else "Pause",
            enabled = if (on) true else s.phase != SendState.Phase.Preparing && (s.test || s.uri != null)) { if (on) s.pause(!s.paused) else s.start() }
    }
    RateSquare(if (on) s.capacityKBs else null)
}

// ai: The lab line (the web's #tx, its first lines): the format, where it is painted, the rate shown and a frame's paint,
// ai: and why not the GPU where auto fell back; for a file its own size and the bytes that go, and which pass shows
// ai: (2026-10-05, the sending line's until then); empty before a first send.
private fun sendLab(s: SendState): String {
    val st = s.stats
    if (st.label.isEmpty()) return ""
    return "${st.label} on the ${if (st.painter == "gpu") "GPU" else "CPU"}, %.1f frames/s, paint %.1f ms a frame".format(Locale.ROOT, st.shownFps, st.paintMs) +
        (if (st.gpuRing) "\nposted ${st.posted}, held {${st.held}}, late ${st.late}, behind ${st.behind}, ${st.filled} of ${st.slots} buffers filled" else "") +
        (if (!s.test && s.size > 0) "\nfile ${s.size} B" + (if (st.sentBytes in 1 until s.size) ", ${st.sentBytes} B as sent (zstd)" else "") +
            (if (st.pass > 0) ", pass ${floor(st.pass).toInt()}" else "") else "") +
        if (st.painter == "cpu" && s.painter != "cpu" && st.gpuWhy.isNotEmpty()) "\nnot the GPU: ${st.gpuWhy}" else ""
}

// ai: Blocks a frame (send.html #blocks and #subchAuto), 0 for auto: the slider's leftmost step is auto (2026-10-05; an
// ai: Auto chip at the title row's end until then), its title "Auto, 60 blocks, 28.1 KB" while what auto takes is known
// ai: ("Auto" before the first configure), every step right of it a count by hand. value the setting, shown what auto
// ai: takes now (0 while unknown), range the counts.
@Composable
private fun AutoSlider(title: String, value: Int, shown: Int, range: IntRange, text: (Int) -> String, choose: (Int) -> Unit) {
    val auto = value == 0
    val b = if (auto) shown else value
    Field(title, if (auto) (if (b > 0) "Auto, ${text(b)}" else "Auto") else text(b)) {
        Bar(value.toFloat(), 0f..range.last.toFloat(), range.last - 1) { choose(it.roundToInt()) }
    }
}

// ai: Pictures a second: a plain slider, 1 to 60, its value beside its title (the web's #fps and #fpsOut)
@Composable
private fun RateSlider(s: SendState) {
    Field("FPS", "${s.rate}") { Bar(s.rate.toFloat(), 1f..60f, 58) { s.chooseRate(it.roundToInt()) } }
}
