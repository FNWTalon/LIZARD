package dev.lizard.receiver

import android.content.Context
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
import kotlin.math.min
import kotlin.math.roundToInt
import java.util.Locale

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
                     val painter: String = "", val gpuWhy: String = "")

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
    var phase by mutableStateOf<Phase>(Phase.Idle)
    // ai: Paused (2026-10-02): nothing presented, the run
    // ai: kept, the last picture left on screen (a receiver holds what it has); a resume presents at once (tick)
    var paused by mutableStateOf(false)
    var stats by mutableStateOf(Stats())

    private var h = 0L
    private var surface: Surface? = null
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
    val capacityKBs get() = if (picked > 0) picked / 8 * 469.0 * rate / 1000 else null
    fun choosePainter(p: String) { painter = p; prefs.edit().putString("sendPainter", p).apply(); if (phase == Phase.On) reconfigure() }

    // ai: the file into the cache (a document is a stream; the sender maps a file), then the sender and its format
    fun start() {
        if (phase == Phase.Preparing || phase == Phase.On) return
        val u = uri
        if (!test && u == null) return
        phase = Phase.Preparing
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
                screenFor(true)
                reconfigure()
                due = 0; lastStats = 0
                vsynced = Build.VERSION.SDK_INT >= 33 && Native.prop("debug.lizard.sendvsync") != "0"
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
        stats = Stats(); paused = false
        screenFor(false)
    }

    // ai: the format the code's square holds now (a turn of the phone re-picks it; the transfer goes on)
    private fun reconfigure() {
        if (h == 0L || room <= 0) return
        val subch = if (blocks > 0) 8 * blocks else Pick.pick(room.toDouble())
        val threads = min(4, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val err = Native.txConfigure(h, Pick.nFor(subch), subch, Pick.span(), fps, threads, when (painter) { "cpu" -> 0; "gpu" -> 1; else -> 2 }, assets)
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
        Log.i(Engine.TAG, "send: LIZARD-$subch${if (blocks > 0) " set by hand" else ""} in a $room px square at $fps a second, painting on the " +
            (if (st?.optString("painter") == "gpu") "GPU (${st.optString("device")})" else "CPU, $threads painters") +
            (st?.optString("gpuWhy").orEmpty().let { if (it.isNotEmpty()) ", not the GPU: $it" else "" }))
    }

    internal val holder = object : SurfaceHolder.Callback {
        override fun surfaceCreated(hd: SurfaceHolder) { surface = hd.surface }
        override fun surfaceChanged(hd: SurfaceHolder, format: Int, w: Int, hh: Int) {
            surface = hd.surface; room = min(w, hh); vw = w; vh = hh
            if (phase == Phase.On && room != configured) reconfigure()
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
            if (Native.txPresent(h, s, vsync, vw, vh)) due = if (due == 0L || t - due > period) t + period else due + period
        }
        if (t - lastStats > 1_000_000_000L) {
            lastStats = t
            runCatching {
                val j = JSONObject(Native.txStats(h))
                stats = Stats(j.optString("label"), j.optDouble("shownFps"), j.optDouble("offeredKBs"), j.optDouble("paintMs"), j.optDouble("pass", 0.0),
                    j.optString("painter"), j.optString("gpuWhy"))
                if (j.optString("error").isNotEmpty()) phase = Phase.Error(j.optString("error"))
            }
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
                val tl = d.preferredFrameTimeline
                tick(tl.expectedPresentationTimeNanos, tl.vsyncId)
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

// ai: Send, as the web sender since 2026-10-02: the file, Start, then Settings (the encoder, the brightness and the
// ai: code: blocks a frame, pictures a second), Developer Tools (the payload, since 2026-10-03; the lab line of the
// ai: last send, shown on the sending screen while it is open)
// ai: and About; the bar fixed over them.
@Composable
internal fun MainActivity.SendScreen() {
    val s = send
    if (s.phase == SendState.Phase.On || s.phase == SendState.Phase.Preparing) { Sending(); return }
    Page("Send", onBack = { go(MainActivity.Screen.Home) }) {
        SectionLabel("File")
        if (s.test) ListRow("Test stream", lead = R.drawable.ic_file, divider = false)
        else if (s.uri == null) ListRow("Choose a file", lead = R.drawable.ic_file, divider = false, onClick = { pickFile() })
        else ListRow(s.name, listOf(if (s.size > 0) Readout.bytes(s.size) else "", s.type).filter { it.isNotEmpty() }.joinToString(" · "),
            lead = R.drawable.ic_file, divider = false, onClick = { pickFile() }) { Btn("Change", Kind.Text) { pickFile() } }
        (s.phase as? SendState.Phase.Error)?.let { Text(it.why, style = MaterialTheme.typography.bodyMedium, color = Bad, modifier = Modifier.padding(top = 12.dp)) }
        Spacer(Modifier.height(16.dp))
        Btn("Start", Kind.Primary, enabled = s.test || s.uri != null, modifier = Modifier.fillMaxWidth()) { s.start() }
        Spacer(Modifier.height(16.dp))
        Fold("Settings", isOpen("sendSettings"), { toggle("sendSettings") }) {
            Fields {
                Field("Encoder") {
                    Chips {
                        Chip("Auto", s.painter == "auto") { s.choosePainter("auto") }
                        Chip("GPU", s.painter == "gpu") { s.choosePainter("gpu") }
                        Chip("CPU", s.painter == "cpu") { s.choosePainter("cpu") }
                    }
                }
                Field("Brightness", "${s.brightness}%") {
                    Bar(s.brightness.toFloat(), 1f..100f, 98) { v -> val n = v.roundToInt(); if (n != s.brightness) s.chooseBrightness(n) }
                }
                Group("Code") {
                    AutoSlider("Blocks a frame", s.blocks, s.picked / 8, 60, 1..128, { n -> "$n block${if (n == 1) "" else "s"}, ${"%.1f".format(Locale.ROOT, n * 469 / 1000.0)}\u00a0KB" }) { s.chooseBlocks(it) }
                    RateSlider(s)
                }
            }
        }
        Fold("Developer Tools", isOpen("sendAdvanced"), { toggle("sendAdvanced") }) {
            Fields {
                Field("Payload") {
                    Chips {
                        Chip("File", !s.test) { s.test = false }
                        Chip("Test stream", s.test) { s.test = true }
                    }
                }
            }
            sendLab(s).let { if (it.isNotEmpty()) CodeBlock(it) }
        }
        About()
    }
}

// ai: The lab line (the web's #tx, its first lines): the format, where it is painted, the rate shown and a frame's paint,
// ai: and why not the GPU where auto fell back; empty before a first send.
private fun sendLab(s: SendState): String {
    val st = s.stats
    if (st.label.isEmpty()) return ""
    return "${st.label} on the ${if (st.painter == "gpu") "GPU" else "CPU"}, %.1f frames/s, paint %.1f ms a frame".format(st.shownFps, st.paintMs) +
        if (st.painter == "cpu" && s.painter != "cpu" && st.gpuWhy.isNotEmpty()) "\nnot the GPU: ${st.gpuWhy}" else ""
}

// ai: Blocks a frame (send.html #blocks and #subchAuto), 0 for auto: its title, the count and what it holds, and the Auto
// ai: chip at the end of the title row, the slider under them the full width (2026-10-02; the chip sat beside the
// ai: slider before). value the setting, shown what auto takes
// ai: now (0 while unknown: no count), dflt the slider's place before either is known; moving the slider sets it by
// ai: hand, the chip goes back to auto (or leaves it at what auto showed).
@Composable
private fun AutoSlider(title: String, value: Int, shown: Int, dflt: Int, range: IntRange, text: (Int) -> String, choose: (Int) -> Unit) {
    val auto = value == 0
    val b = if (auto) shown else value
    Field(title, if (b > 0) text(b) else "", end = { Chip("Auto", auto) { choose(if (auto) (if (b > 0) b else dflt) else 0) } }) {
        Bar((if (b > 0) b else dflt).toFloat(), range.first.toFloat()..range.last.toFloat(), range.last - range.first - 1) { choose(it.roundToInt()) }
    }
}

// ai: Pictures a second: a plain slider, 1 to 60, its value beside its title (the web's #fps and #fpsOut)
@Composable
private fun RateSlider(s: SendState) {
    Field("Pictures a second", "${s.rate} a second") { Bar(s.rate.toFloat(), 1f..60f, 58) { s.chooseRate(it.roundToInt()) } }
}

// ai: The code as large as the screen allows on white, and under it (beside it, turned) the state, the rate and Stop.
@Composable
private fun MainActivity.Sending() {
    val s = send
    BoxWithConstraints(Modifier.fillMaxSize().background(Bg).safeDrawingPadding().padding(12.dp)) {
        val land = maxWidth > maxHeight
        // ai: the panel beside the code in landscape, or its collapser alone (sendCollapsed, 2026-10-02): the code the
        // ai: larger where the width bounds it
        val folded = land && isOpen("sendCollapsed")
        val side = if (land) min(maxHeight.value, maxWidth.value - (if (folded) 72f else 280f)) else min(maxWidth.value, maxHeight.value - 150f)
        val code: @Composable () -> Unit = {
            AndroidView({ ctx -> SurfaceView(ctx).apply { holder.addCallback(s.holder) } }, Modifier.size(side.dp))
        }
        val panel: @Composable () -> Unit = {
            Column(Modifier.padding(horizontal = 4.dp)) {
                val what = if (s.test) "the test stream" else s.name
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(if (s.phase == SendState.Phase.Preparing) "Preparing $what" else if (s.paused) "Paused" else "Sending $what", style = MaterialTheme.typography.titleMedium, color = Fg,
                        modifier = Modifier.weight(1f).padding(top = 8.dp))
                    if (land) CollapseBtn(false, left = false) { toggle("sendCollapsed") }
                }
                val st = s.stats
                if (st.offeredKBs > 0) Text(Readout.rate(st.offeredKBs) + if (!s.test && st.pass > 0) ", pass ${floor(st.pass).toInt()}" else "",
                    style = MaterialTheme.typography.bodyMedium, color = Muted)
                HeatWarning(heat, clocks, Modifier.padding(top = 8.dp))
                if (isOpen("sendAdvanced")) sendLab(s).let { if (it.isNotEmpty()) CodeBlock(it) }
                Spacer(Modifier.height(8.dp))
                if (s.paused) { Btn("Resume", Kind.Primary, modifier = Modifier.fillMaxWidth()) { s.pause(false) }; Spacer(Modifier.height(8.dp)) }
                Btn("Stop", modifier = Modifier.fillMaxWidth()) { s.stop() }
            }
        }
        if (land) Row(Modifier.fillMaxSize(), horizontalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterHorizontally), verticalAlignment = Alignment.CenterVertically) {
            code()
            // ai: the rail (as Receive's): pause or resume, and the code's capacity
            if (folded) Rail(left = false, onClick = { toggle("sendCollapsed") }) {
                Square { IconBtn(if (s.paused) R.drawable.ic_play else R.drawable.ic_pause, if (s.paused) "Resume" else "Pause", enabled = s.phase == SendState.Phase.On) { s.pause(!s.paused) } }
                RateSquare(s.capacityKBs)
            }
            else Box(Modifier.width(260.dp)) { panel() }
        } else Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally) { code(); panel() }
    }
}
