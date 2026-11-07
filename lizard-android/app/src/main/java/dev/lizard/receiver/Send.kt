package dev.lizard.receiver

import dev.lizard.pick.Pick
import dev.lizard.readout.Readout
import android.content.Context
import android.content.res.Configuration
import androidx.compose.ui.unit.Dp
import androidx.compose.material3.Icon
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.style.TextAlign
import android.net.Uri
import android.os.Build
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import dev.lizard.tar.Bundle
import dev.lizard.tar.TAR_TYPE
import dev.lizard.tar.TarEntry
import android.util.Log
import android.view.Choreographer
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import androidx.annotation.RequiresApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
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
import dev.lizard.ui.*
import androidx.compose.ui.unit.DpSize

// ai: Send (2026-10-01): the web sender's job on the phone, the C on the CPU or the GPU painting (`painter` below;
// ai: liblizard/core/tx/sender.h). A file chosen here or shared to LIZARD from another app is copied into the cache and
// ai: mapped by the sender; the
// ai: format is the largest the code's square holds (Pick, the web's pickVersion), in the default ring, at Settings'
// ai: FPS (60 unless set; until 2026-10-02 "Receiving with" named it, the app 60, a browser 24); frames go onto a
// ai: SurfaceView at that pace (a Choreographer callback each vsync posting the next painted frame when one is due; the
// ai: surface asks the display for that rate, the switch always made), the screen kept on, at the Brightness set (the phone's own unless moved), its
// ai: bars hidden. The test stream is Developer Tools' (Payload). Vsync-locked since 2026-10-02: on
// ai: Android 13 and up each frame is due by its frame timeline's expected presentation time and posted for that vsync
// ai: (jni.cpp GpuRing, Ring), not into the window's queue at whatever refresh it is latched for; below 13, and for the
// ai: rest of a send where the device or the surface refuses that present (toWindowQueue), the window's queue.
class SendState(private val a: MainActivity) {
    sealed interface Phase { data object Idle : Phase; data object Preparing : Phase; data object On : Phase; data class Error(val why: String) : Phase }
    data class Stats(val label: String = "", val shownFps: Double = 0.0, val offeredKBs: Double = 0.0, val paintMs: Double = 0.0, val pass: Double = 0.0,
                     val painter: String = "", val gpuWhy: String = "", val sentBytes: Long = 0,   // ai: sentBytes: the file's bytes as they go (2026-10-05)
                     // ai: the GPU ring's tally over the last second (2026-10-07, jni.cpp txPostStats): posted, held by
                     // ai: vsyncs, late, behind, buffers filled of the ring's
                     val gpuRing: Boolean = false, val posted: Long = 0, val held: String = "", val late: Long = 0, val behind: Long = 0, val filled: Int = 0, val slots: Int = 0,
                     // ai: the blocks a code carries (2026-10-07)
                     val blocks: Int = 0)

    private val prefs = a.getSharedPreferences("lizard", Context.MODE_PRIVATE)
    // ai: What Start sends (2026-10-16): one file as itself, or several files or a folder as one tar
    // ai: (liblizard/kotlin/Tar.kt Bundle, the web sender's rule); `reading` while a pick is walked and sized
    var choice by mutableStateOf<Bundle?>(null)
    var reading by mutableStateOf(false)
    var name by mutableStateOf("")
    var size by mutableStateOf(0L)
    var type by mutableStateOf("")
    val ready get() = choice != null && !reading
    // ai: The transfer made at Start (2026-11-01; at the pick and again at each stop from 2026-10-19, every chunk
    // ai: compressed for picks never sent): the file copied (or the bundle's tar written) into the cache and every chunk
    // ai: compressed (txCreate) on a thread of its own, the line "Preparing" meanwhile. compressed: its bytes as they go,
    // ai: 0 until a run makes it, kept through the run and after it until the choice changes.
    var compressed by mutableStateOf(0L)
    private var made = 0L                 // ai: the transfer made for the choice, not yet sending (0: none)
    private var madeFile: File? = null    // ai: the cache file it maps
    private var runFile: File? = null     // ai: the one the run's transfer maps
    private var making = 0                // ai: the make under way's number: a newer make or pick orphans it
    private var makingNow = false
    private var startWhenMade = false     // ai: Start pressed while the transfer was being made
    var test by mutableStateOf(false)
    // ai: The format's size (2026-10-02), the web sender's
    // ai: slider (send.html #blocks and #subchAuto): 0 for auto, the largest the code's square holds (Pick.pick), else a
    // ai: version 1 to 64 set by hand (16 sub-channels a step, Pick.STEP), painted as set whatever the square (the web's
    // ai: too); kept, as the web's #subch is ("sendVersion" since the versions went to 16 sub-channels, 2026-10-29: the
    // ai: "sendBlocks" before counted 8). picked: the sub-channels the last configure took, which auto's label names.
    var blocks by mutableStateOf(prefs.getInt("sendVersion", 0).let { if (it in Pick.VERSION_MIN..Pick.VERSION_MAX) it else 0 })
    var picked by mutableStateOf(0)
    // ai: VSync (2026-10-23, on by default, the desktop sender's): the rate voted on both surfaces (the window's and the
    // ai: ring's, jni.cpp), so the panel moves to a mode the rate divides; off, no vote, the system's choice. Kept.
    var vsyncOn by mutableStateOf(prefs.getBoolean("sendVsync", true))
    // ai: Pictures a second (2026-10-02), the web's #fps: 60 unless set (the word states it; a rate the display's refresh
    // ai: does not divide holds pictures unevenly); kept, as the web's is. "Receiving with" (the app 60, a browser 24) went
    // ai: the same day. Under VSync Pick.FPS's chips (2026-10-23; a kept rate read as the nearest), without it a 1 to 60
    // ai: slider; VSync turned on takes a slider's rate to the nearest of the chips'.
    var rate by mutableStateOf(prefs.getInt("sendFps", 60).let { if (vsyncOn) Pick.fpsOf(it) else it.coerceIn(1, 60) })
    // ai: The screen's brightness while sending (2026-10-02): 0 the phone's own (System, at every launch since
    // ai: 2026-10-27: forcing 100 at each Start, the default from 2026-10-03, was a nuisance and did not always hold), or
    // ai: 1 to 100% once the slider is moved, the window's own override, so the phone's setting is left as it was once
    // ai: sending stops; set at Start (screenFor), the settings being the idle screen's.
    var brightness by mutableStateOf(0)
    private val brightnessOverride get() = if (brightness > 0) brightness / 100f else WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
    // ai: Painting with (2026-10-02; the web's encoder switch): auto (the GPU
    // ai: where it builds and its first frames are the C's, else the CPU), the GPU (liblizard/core/tx/gpu_painter.h) or the CPU's
    // ai: painters; kept.
    var painter by mutableStateOf(prefs.getString("sendPainter", "auto") ?: "auto")
    // ai: Codes (2026-10-05, the web's #codes and #gap): one, or two side by side for a receiver's 2:1 crop, `gap` modules
    // ai: apart (0 to 64, 12 unless set); kept. The format is picked for one code's share of the box (roomOf).
    var codes by mutableStateOf(prefs.getInt("sendCodes", 1).coerceIn(1, 2))
    var gap by mutableStateOf(prefs.getInt("sendGap", 12).coerceIn(0, 64))
    // ai: the ring (2026-10-31, as the desktop's): an index into Pick.RINGS, -1 the default (the 128); kept
    var ring by mutableStateOf(prefs.getInt("sendRing", -1).let { if (it in Pick.RINGS.indices) it else -1 })
    val ringIndex get() = if (ring in Pick.RINGS.indices) ring else Pick.RING_DEFAULT
    // ai: Margin (2026-10-27; the web's and the desktop's since 2026-10-23):
    // ai: the light between a code's ring and the box's edge, modules, its quiet zone (Pick.QUIET) at least and
    // ai: MARGIN_MAX at most, MARGIN_DEFAULT unless set; kept (sendMargin). Not painted: the box keeps it round the code's
    // ai: surface (CodeBox), as the desktop's presenter keeps it round the frame.
    var margin by mutableStateOf(prefs.getInt("sendMargin", MARGIN_DEFAULT).coerceIn(Pick.QUIET, MARGIN_MAX))
    private val sideModules get() = Pick.modules(ringIndex) + 2.0 * Pick.QUIET   // ai: a painted symbol's side, its quiet zone in
    private val past get() = (margin - Pick.QUIET).toDouble()           // ai: the margin past that quiet zone, modules
    // ai: the painted frame's width over its height: the codes and the gaps between them, each code as tall as the frame
    // ai: (a gap of `gap` modules a share of one code's side: the web's gapFrac)
    private val frameAspect get() = codes + (codes - 1) * gap / sideModules
    // ai: the code box's: the frame and the margin past its quiet zone at each edge
    val aspect get() = (frameAspect * sideModules + 2 * past) / (sideModules + 2 * past)
    // ai: that margin a share of the box's height (CodeBox's padding round the surface)
    val edgeShare get() = past / (sideModules + 2 * past)
    var phase by mutableStateOf<Phase>(Phase.Idle)
    // ai: Paused (2026-10-02): nothing presented, the run
    // ai: kept, the last picture left on screen (a receiver holds what it has); a resume presents at once (tick)
    var paused by mutableStateOf(false)
    var stats by mutableStateOf(Stats())

    private var h = 0L
    private var surface: Surface? = null
    var shown by mutableStateOf(false)   // ai: a picture has been presented on this surface (Send.kt CodeBox's overlay)
    private var placed = false           // ai: and placed for the surface's current size (a post sets the ring's geometry)
    private var room = 0
    private var configured = 0          // ai: the room the sender was last configured for
    private var due = 0L
    private var lastStats = 0L
    private var vsynced = false         // ai: this send's frames posted for their vsync (Android 13 on, unless refused: toWindowQueue)
    var refused by mutableStateOf("")   // ai: why the vsync present was refused in this send, for the lab line ("" none)
    private var assets = ""             // ai: the generated tree (Assets.ensure), the GPU painter's kernels and tables
    private var vw = 0                  // ai: the view's size in pixels, the vsync-locked present's destination
    private var vh = 0
    val fps get() = rate

    // ai: The selection (2026-10-23; until then a pick replaced the last): items added by each pick, share or photo pick,
    // ai: in order, each a file or a folder with its entries under its own name, so files and folders mix; the choice is
    // ai: their Bundle (one file alone is itself, else one tar). An item already listed (its document's URI) is not
    // ai: added again. A walk runs off the main thread, and one that fails leaves the list as it was and says why. Every
    // ai: change stops a run first, then the bundle is made off the main thread and its transfer made again (make), as
    // ai: the web's does.
    class Item(val key: String, val name: String, val dir: Boolean, val files: Int, val size: Long, val entries: List<TarEntry>)
    val items = mutableStateListOf<Item>()
    fun pick(us: List<Uri>) = add { us.map { u -> fileEntry(u).let { e -> Item(u.toString(), e.path, false, 1, e.size, listOf(e)) } } }
    fun pickTree(tree: Uri) = add {
        val es = treeEntries(tree)
        if (es.isEmpty()) emptyList() else listOf(Item(tree.toString(), es[0].path, true, es.count { !it.dir }, es.filter { !it.dir }.sumOf { it.size }, es))
    }
    fun remove(key: String) { items.removeAll { it.key == key }; changed() }
    fun clear() { items.clear(); changed() }
    private fun add(walk: () -> List<Item>) {
        if (phase == Phase.On || phase == Phase.Preparing) stop()
        test = false; reading = true
        if (phase is Phase.Error) phase = Phase.Idle
        thread(name = "lizard-send-pick") {
            val got = try { walk() } catch (e: Exception) { a.runOnUiThread { reading = bundling != bundled; phase = Phase.Error("The files could not be read: ${e.message}") }; return@thread }
            a.runOnUiThread {
                reading = bundling != bundled
                val have = items.mapTo(HashSet()) { it.key }
                val fresh = got.filter { have.add(it.key) }
                if (fresh.isNotEmpty()) { items.addAll(fresh); changed() }
            }
        }
    }
    private var bundling = 0     // ai: the newest bundle asked for, and the newest one made: equal while none is being made
    private var bundled = 0
    private fun changed() {
        if (phase == Phase.On || phase == Phase.Preparing) stop()
        dropMade()
        if (phase is Phase.Error) phase = Phase.Idle
        choice = null; name = ""; size = 0; type = ""; compressed = 0
        val list = items.toList()
        typeOf.keys.retainAll(list.flatMap { it.entries }.toSet())
        val my = ++bundling
        if (list.isEmpty()) { bundled = my; reading = false; return }
        reading = true
        thread(name = "lizard-send-bundle") {
            val b = Bundle(list.flatMap { it.entries })
            a.runOnUiThread {
                if (my != bundling) return@runOnUiThread
                bundled = my; reading = false; choice = b
                name = b.name; size = b.size; type = b.single?.let { typeOf[it] } ?: TAR_TYPE
            }
        }
    }
    private val typeOf = HashMap<TarEntry, String>()   // ai: a single file's own media type, the header's
    // ai: A document's name, size and time; a provider that says no size is read through once to count it (a tar
    // ai: header needs the exact size before the bytes).
    private fun fileEntry(u: Uri): TarEntry {
        val cr = a.contentResolver
        var nm = "file"; var sz = -1L; var t: Long? = null
        cr.query(u, null, null, null, null)?.use { c ->
            if (c.moveToFirst()) {
                c.getColumnIndex(OpenableColumns.DISPLAY_NAME).takeIf { it >= 0 && !c.isNull(it) }?.let { nm = c.getString(it) }
                c.getColumnIndex(OpenableColumns.SIZE).takeIf { it >= 0 && !c.isNull(it) }?.let { sz = c.getLong(it) }
                c.getColumnIndex(DocumentsContract.Document.COLUMN_LAST_MODIFIED).takeIf { it >= 0 && !c.isNull(it) }?.let { t = c.getLong(it) / 1000 }
            }
        }
        if (sz < 0) sz = cr.openInputStream(u)!!.use { i -> var n = 0L; val b = ByteArray(1 shl 16); while (true) { val k = i.read(b); if (k < 0) break; n += k }; n }
        return TarEntry(plainName(nm), sz, t) { cr.openInputStream(u) ?: throw java.io.IOException("$nm could not be opened") }.also { typeOf[it] = cr.getType(u) ?: "" }
    }
    // ai: A folder chosen through the system's tree picker, walked by DocumentsContract's child queries: its folders
    // ai: (an empty one too) and files under its own name. 64 levels at most, so a provider's loop cannot run on.
    private fun treeEntries(tree: Uri): List<TarEntry> {
        val cr = a.contentResolver
        val cols = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE, DocumentsContract.Document.COLUMN_SIZE, DocumentsContract.Document.COLUMN_LAST_MODIFIED)
        val out = ArrayList<TarEntry>()
        fun time(c: android.database.Cursor) = if (c.isNull(4)) null else c.getLong(4) / 1000
        fun walk(id: String, path: String, t: Long?, depth: Int) {
            if (depth > 64) throw java.io.IOException("$path: too deeply nested")
            out.add(TarEntry(path, mtimeSec = t))
            val dirs = ArrayList<Triple<String, String, Long?>>()
            cr.query(DocumentsContract.buildChildDocumentsUriUsingTree(tree, id), cols, null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    val kid = c.getString(0); val nm = plainName(c.getString(1) ?: kid)
                    if (c.getString(2) == DocumentsContract.Document.MIME_TYPE_DIR) { dirs.add(Triple(kid, nm, time(c))); continue }
                    val u = DocumentsContract.buildDocumentUriUsingTree(tree, kid)
                    val sz = if (c.isNull(3)) -1L else c.getLong(3)
                    out.add(if (sz >= 0) TarEntry("$path/$nm", sz, time(c)) { cr.openInputStream(u) ?: throw java.io.IOException("$nm could not be opened") }
                            else fileEntry(u).let { TarEntry("$path/$nm", it.size, time(c), it.open) })
                }
            }
            for ((kid, nm, kt) in dirs) walk(kid, "$path/$nm", kt, depth + 1)
        }
        val rootId = DocumentsContract.getTreeDocumentId(tree)
        cr.query(DocumentsContract.buildDocumentUriUsingTree(tree, rootId), cols, null, null, null)?.use { c ->
            if (c.moveToFirst()) walk(rootId, plainName(c.getString(1) ?: "folder"), time(c), 0)
        }
        return out
    }
    // ai: a provider's name as one part of a path: no separator, none empty
    private fun plainName(n: String) = n.replace('/', '_').ifEmpty { "file" }
    // ai: a slider's value is kept as it moves and applied to a running send when it is let go (settle): each
    // ai: configure is a new format, its first frames tens of ms away, and one a drag step kept the code flashing
    private var unsettled = false
    fun settle() { if (unsettled) { unsettled = false; if (phase == Phase.On) reconfigure() } }
    fun chooseBlocks(b: Int) { val n = if (b in Pick.VERSION_MIN..Pick.VERSION_MAX) b else 0; if (n == blocks) return; blocks = n; prefs.edit().putInt("sendVersion", blocks).apply(); unsettled = true }
    fun chooseRate(r: Int, now: Boolean = true) { rate = r.coerceIn(1, 60); prefs.edit().putInt("sendFps", rate).apply(); unsettled = true; if (now) settle() }
    fun chooseVsync(on: Boolean) {
        vsyncOn = on
        if (on && rate !in Pick.FPS) rate = Pick.fpsOf(rate)
        prefs.edit().putBoolean("sendVsync", on).putInt("sendFps", rate).apply()
        if (phase == Phase.On) reconfigure()
    }
    // ai: the window holds the brightness while sending (screenFor), so a move mid-send is applied at once
    fun chooseBrightness(b: Int) {
        brightness = if (b <= 0) 0 else b.coerceAtMost(100)
        if (phase == Phase.On) a.window.let { w -> w.attributes = w.attributes.apply { screenBrightness = brightnessOverride } }
    }
    fun pause(p: Boolean) { if (phase == Phase.On) paused = p }
    // ai: what the code offers at most, KB/s: blocks a frame x 472 B x pictures a second; null before the first configure
    val capacityKBs get() = if (picked > 0) codes * (if (stats.blocks > 0) stats.blocks else Native.blocksFor(picked)) * Pick.PAYLOAD.toDouble() * rate / 1000 else null
    fun choosePainter(p: String) { painter = p; prefs.edit().putString("sendPainter", p).apply(); if (phase == Phase.On) reconfigure() }
    fun chooseCodes(c: Int) { codes = c.coerceIn(1, 2); prefs.edit().putInt("sendCodes", codes).apply(); if (phase == Phase.On) reconfigure() }
    fun chooseRing(v: Int) { ring = if (v in Pick.RINGS.indices) v else -1; prefs.edit().putInt("sendRing", ring).apply(); if (phase == Phase.On) reconfigure() }
    fun chooseGap(g: Int) { gap = g.coerceIn(0, 64); prefs.edit().putInt("sendGap", gap).apply(); unsettled = true }
    // ai: at the slider's release (the box resized once: a resize mid-send reconfigures where the room changed)
    fun chooseMargin(m: Int) { margin = m.coerceIn(Pick.QUIET, MARGIN_MAX); prefs.edit().putInt("sendMargin", margin).apply() }
    // ai: one code's room in the view, px: the box's height, or its width's share where that is less
    private fun roomOf() = min((vw / frameAspect).toInt(), vh)   // ai: the surface holds the frame alone
    // ai: the column's edge let go (2026-10-05): the format re-picked for the box it left
    fun resized() { if (phase == Phase.On && roomOf() != configured) reconfigure() }

    // ai: The choice's transfer made off the main thread: the file into the cache (a document is a stream; the sender maps
    // ai: a file), or the bundle's tar written there from each document in turn, then txCreate (each chunk compressed,
    // ai: its fountain set up) and its bytes as sent from the stats. The cache keeps the made one's file and the run's.
    private fun make() {
        val b = choice ?: return
        dropMade()
        val my = making
        makingNow = true
        val nm = name; val ty = type
        val keep = runFile?.name
        thread(name = "lizard-send-make") {
            val dir = File(a.cacheDir, "send").apply { mkdirs() }
            dir.listFiles()?.forEach { if (it.name != keep) it.delete() }
            val f = File(dir, "file-$my")
            val wrote = try {
                val one = b.single
                if (one != null) one.open!!().use { i -> f.outputStream().use { o -> i.copyTo(o, 1 shl 20) } }
                else f.outputStream().buffered(1 shl 20).use { o -> b.write(o) }
                null
            } catch (e: Exception) { e }
            val hnd = if (wrote != null) 0L else Native.txCreate(f.path, nm, ty)
            val err = if (wrote != null) "The file could not be read: ${wrote.message}" else if (hnd == 0L) "This file cannot be sent: ${Native.txError()}" else ""
            val sent = if (hnd != 0L) runCatching { JSONObject(Native.txStats(hnd)).optLong("sentBytes", 0) }.getOrDefault(0L) else 0L
            a.runOnUiThread {
                if (my != making) { if (hnd != 0L) Native.txDestroy(hnd); f.delete(); return@runOnUiThread }   // ai: let go meanwhile
                makingNow = false
                if (hnd == 0L) { f.delete(); startWhenMade = false; phase = Phase.Error(err); return@runOnUiThread }
                made = hnd; madeFile = f; compressed = sent
                if (startWhenMade) { startWhenMade = false; go() }
            }
        }
    }
    // ai: the made transfer let go, and any make under way orphaned
    private fun dropMade() {
        making++; makingNow = false; startWhenMade = false
        if (made != 0L) { Native.txDestroy(made); made = 0L }
        madeFile?.delete(); madeFile = null
    }

    fun start() {
        if (phase == Phase.Preparing || phase == Phase.On) return
        val b = choice
        if (!test && (b == null || reading)) return
        phase = Phase.Preparing
        shown = false; placed = false
        // ai: the file's transfer still being made (or its make let go): Start waits for it
        if (!test && made == 0L) { if (!makingNow) make(); startWhenMade = true; return }
        go()
    }
    // ai: Sending begins: the made transfer taken (the test stream's made here), the GPU painter's device and pipelines
    // ai: made off the main thread, then the format and the posting
    private fun go() {
        val pre = if (test) 0L else made
        if (!test) { made = 0L; runFile = madeFile; madeFile = null }
        val my = making
        thread(name = "lizard-send-prep") {
            val hnd = if (pre != 0L) pre else Native.txCreate("", "", "")
            val err = if (hnd == 0L) Native.txError() else ""
            // ai: the GPU painter's device and pipelines made here, off the main thread that configures
            if (hnd != 0L && painter != "cpu") {
                assets = Assets.ensure(a).path
                val why = Native.txPrepare(hnd, assets)
                if (why.isNotEmpty()) Log.i(TAG, "send: no GPU painter: $why")
            }
            a.runOnUiThread {
                if (hnd == 0L) { phase = Phase.Error("This file cannot be sent: $err"); return@runOnUiThread }
                if (phase != Phase.Preparing || my != making) { Native.txDestroy(hnd); return@runOnUiThread }   // ai: stopped meanwhile
                h = hnd; configured = 0; phase = Phase.On
                vsynced = Build.VERSION.SDK_INT >= 33
                refused = ""
                screenFor(true)
                reconfigure()
                due = 0; lastStats = 0; lastTimeline = 0; vsyncNs = 0
                Log.i(TAG, "send: ${if (vsynced) "frames posted for their vsync" else "frames into the window's queue"}")
                if (vsynced && Build.VERSION.SDK_INT >= 33) Choreographer.getInstance().postVsyncCallback(vsync)
                else Choreographer.getInstance().postFrameCallback(frame)
            }
        }
    }

    fun stop() {
        Choreographer.getInstance().removeFrameCallback(frame)
        if (Build.VERSION.SDK_INT >= 33) Choreographer.getInstance().removeVsyncCallback(vsync)
        if (h != 0L) { Native.txDestroy(h); h = 0 }
        runFile?.delete(); runFile = null
        startWhenMade = false
        if (phase == Phase.On || phase == Phase.Preparing) phase = Phase.Idle
        shown = false; placed = false
        stats = Stats(); paused = false
        screenFor(false)
    }

    // ai: the format the code's square holds now (a turn of the phone re-picks it; the transfer goes on)
    private fun reconfigure() {
        room = roomOf()
        if (h == 0L || room <= 0) return
        val subch = if (blocks > 0) Pick.STEP * blocks else Pick.pick(room.toDouble(), ringIndex)
        val threads = min(4, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val err = Native.txConfigure(h, subch, Pick.span(ringIndex), fps, threads, when (painter) { "cpu" -> 0; "gpu" -> 1; else -> 2 }, assets, codes, gap, vsynced, vsyncOn)
        if (err.isNotEmpty()) { phase = Phase.Error(err); stop(); return }
        configured = room
        picked = subch
        // ai: the window's queue's vote (the vsync-locked present votes on its own surface, jni.cpp): the switch always
        // ai: made from Android 12 (the default makes it only where seamless, which can leave the panel at another rate);
        // ai: VSync off, 0, no vote
        runCatching {
            val vote = if (vsyncOn) fps.toFloat() else 0f
            if (Build.VERSION.SDK_INT >= 31) surface?.setFrameRate(vote, Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE, Surface.CHANGE_FRAME_RATE_ALWAYS)
            else if (Build.VERSION.SDK_INT >= 30) surface?.setFrameRate(vote, Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE)
        }
        val st = runCatching { JSONObject(Native.txStats(h)) }.getOrNull()
        Log.i(TAG, "send: LIZARD-$subch${if (codes > 1) " x $codes, $gap modules apart" else ""}${if (blocks > 0) " set by hand" else ""} in a $room px square at $fps a second, painting on the " +
            (if (st?.optString("painter") == "gpu") "GPU (${st.optString("device")})" else "CPU, $threads painters") +
            (st?.optString("gpuWhy").orEmpty().let { if (it.isNotEmpty()) ", not the GPU: $it" else "" }))
    }

    internal val holder = object : SurfaceHolder.Callback {
        override fun surfaceCreated(hd: SurfaceHolder) { surface = hd.surface }
        override fun surfaceChanged(hd: SurfaceHolder, format: Int, w: Int, hh: Int) {
            if (w != vw || hh != vh) placed = false   // ai: a picture placed for this size wanted, paused or not
            surface = hd.surface; vw = w; vh = hh
            // ai: not while the column's edge is being dragged: the code's box changes every frame of it (resized after)
            if (phase == Phase.On && roomOf() != configured && !a.sideDragging) reconfigure()
        }
        // ai: the view detached (the code's box moved between full screen and the page, 2026-10-27) destroys its surface,
        // ai: and the ring's child surface went with it while posts went on into it, the code black on the page
        // ai: the next surface shows a picture even while paused (tick, tickVsync), the file's look over it until then
        override fun surfaceDestroyed(hd: SurfaceHolder) { surface = null; shown = false; placed = false; if (h != 0L) Native.txSurfaceGone(h) }
    }

    // ai: each frame callback (the window's queue): the next painted frame when one is due at the asked rate (half a vsync
    // ai: early is on time). t: the callback's frame time, the window's queue showing it a refresh or two after.
    private fun tick(t: Long) {
        val period = 1_000_000_000L / fps
        val s = surface
        // ai: paused: presented at once on a resume; a surface with no picture placed for its size (the paused page's, or
        // ai: one resized under it: the column folded) takes one
        if (paused && placed) due = 0L
        else if (s != null && configured > 0 && t >= due - 4_000_000L) {
            if (Native.txPresent(h, s)) { due = if (due == 0L || t - due > period) t + period else due + period; shown = true; placed = true }
        }
        pollStats(t)
    }
    // ai: The vsync path's schedule (2026-10-07: a fixed schedule, every picture on its due refresh): the display's period
    // ai: from the frame timelines' spacing; a picture every round(period / asked) vsyncs where that is whole within
    // ai: 1%, else on the asked rate's grid (the web's due rule); for every timeline the Choreographer offers (the
    // ai: preferred one and the ones after it) not yet passed, the next filled buffer is posted for it where a picture
    // ai: is due there (Native.txPost), so the compositor holds the pictures ahead. Nothing filled at a due vsync: the
    // ai: picture on screen stays (counted behind in the ring's tally) and the next picture takes the next due vsync. A
    // ai: refusal (no GPU ring, child surface or buffers here, or the ring's filler failed) takes the window's queue.
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
        // ai: paused, as the window's queue (tick): nothing posted once this surface shows a picture placed for its size
        if (paused && placed) { due = 0L; lastTimeline = tls.lastOrNull()?.expectedPresentationTimeNanos ?: lastTimeline }
        else for (tl in tls) {
            if (paused && placed) break
            val t = tl.expectedPresentationTimeNanos
            if (t <= lastTimeline) continue
            lastTimeline = t
            if (s == null || configured <= 0 || t + vsyncNs / 2 < due) continue
            when (Native.txPost(h, s, tl.vsyncId, t, vsyncNs, vw, vh)) {
                1 -> { due = if (due == 0L || t - due > periodNs) t + periodNs else due + periodNs; shown = true; placed = true }
                0 -> due = t + periodNs
                else -> { toWindowQueue(runCatching { JSONObject(Native.txPostStats(h)).optString("error") }.getOrDefault("")); return }
            }
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
                p?.optBoolean("gpu") ?: false, p?.optLong("posted") ?: 0, held, p?.optLong("late") ?: 0, p?.optLong("behind") ?: 0, p?.optInt("filled") ?: 0, p?.optInt("slots") ?: 0,
                j.optInt("blocks", 0))
            if (p?.optBoolean("gpu") == true) Log.i(TAG, "send: posted ${stats.posted}, held {${held}}, late ${stats.late}, behind ${stats.behind}, ${stats.filled} of ${stats.slots} buffers filled, ${j.optInt("ahead")} of ${j.optInt("depth")} frames painted ahead")
            if (j.optString("error").isNotEmpty()) phase = Phase.Error(j.optString("error"))
            // ai: the bytes as they go, known once every chunk is made (the core streams: its first frames go while the
            // ai: chunks are made, 2026-11-06), so the size line takes them from here
            if (compressed == 0L && stats.sentBytes > 0) compressed = stats.sentBytes
        }
    }
    // ai: The vsync present refused for this send (jni.cpp txPost -1; its child surface let go): the frames go into the
    // ai: window's own queue from here, by the frame callback, the painters keeping host frames again (the configure
    // ai: with vsynced off). Why goes to the log and the lab line.
    private fun toWindowQueue(why: String) {
        if (!vsynced) return
        Log.i(TAG, "send: the vsync present refused (${why.ifEmpty { "no reason given" }}): frames into the window's queue")
        refused = why.ifEmpty { "refused" }
        vsynced = false
        if (Build.VERSION.SDK_INT >= 33) Choreographer.getInstance().removeVsyncCallback(vsync)
        due = 0
        reconfigure()
        if (h != 0L) Choreographer.getInstance().postFrameCallback(frame)
    }
    private val frame = object : Choreographer.FrameCallback {
        override fun doFrame(t: Long) {
            if (h == 0L) return
            tick(t)
            Choreographer.getInstance().postFrameCallback(this)
        }
    }
    // ai: Android 13 on: the preferred frame timeline, the one the system expects a frame started now to make
    private val vsync: Choreographer.VsyncCallback by lazy @RequiresApi(33) {
        object : Choreographer.VsyncCallback {
            override fun onVsync(d: Choreographer.FrameData) {
                if (h == 0L || !vsynced) return
                tickVsync(d)
                if (vsynced) Choreographer.getInstance().postVsyncCallback(this)
            }
        }
    }

    // ai: while sending: the screen on, at the Brightness set, its bars hidden; all put back after
    private fun screenFor(on: Boolean) {
        val w = a.window
        val c = WindowCompat.getInsetsController(w, w.decorView)
        if (on) {
            w.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            w.attributes = w.attributes.apply { screenBrightness = brightnessOverride }
            c.hide(WindowInsetsCompat.Type.systemBars())
        } else {
            w.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            w.attributes = w.attributes.apply { screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE }
            c.show(WindowInsetsCompat.Type.systemBars())
        }
    }
}

// ai: Send on the phone (the screen is ../lizard-ui's SendScreen): the sender's state (SendState), the phone's settings
// ai: (Brightness, no Fullscreen: a send running is the whole screen, a tap at the left edge its rail, Back stops it),
// ai: the code's box (CodeBox: the code painted into a SurfaceView, or the upload in Compose), the power-saving question
internal class AndroidSend(private val a: MainActivity) : SendHost {
    private val s get() = a.send
    override val folds = a.receiveUi.folds
    override val width = a.receiveUi.width
    override val canPortrait = true
    override val swapInColumn = false
    override fun goHome() = a.go(Screen.Home)
    override fun goReceive() = a.go(Screen.Receive)
    override val on get() = s.phase == SendState.Phase.On
    override val busy get() = on || s.phase == SendState.Phase.Preparing
    override val paused get() = s.paused
    override val canStart get() = s.phase != SendState.Phase.Preparing && (s.test || s.ready)
    override fun start() = s.start()
    override fun stop() = s.stop()
    override fun pause(on: Boolean) = s.pause(on)
    override val pauseShown get() = s.paused
    override val fullscreen = false
    override fun toggleFull() {}
    // ai: the state line: the file and its sizes ("<compressed> (<raw> raw)" once a run has made the transfer, the size
    // ai: alone before), "Preparing", "Sending" with the bytes that go (paused too: Resume tells a pause, no "Paused" line,
    // ai: 2026-11-06), or what went wrong
    override val line: SendLine get() {
        val what = s.name
        val count = s.choice?.takeIf { it.single == null }?.files?.let { "$it ${if (it == 1) "file" else "files"}, " } ?: ""
        val sizeText = if (s.compressed > 0) "${Readout.bytes(s.compressed)} (${Readout.bytes(s.size)}\u00a0raw)" else Readout.bytes(s.size)
        val err = (s.phase as? SendState.Phase.Error)?.why
        val preparing = s.phase == SendState.Phase.Preparing
        val named = !s.test && err == null && (preparing || on || (!on && !s.reading && s.choice != null))
        if (named) return SendLine(when { preparing -> "Preparing "; on -> "Sending "; else -> "" }, what,
            if (preparing || s.size <= 0) "" else ", ${if (on) "" else count}$sizeText", false)
        val sized = if (s.size <= 0) what else "$what, ${if (on) "" else count}$sizeText"
        val text = when {
            err != null -> err
            s.test -> if (preparing) "Preparing" else if (on) "Sending" else ""
            preparing -> "Preparing $what"
            on -> "Sending $sized"
            s.reading -> "Reading the files"
            s.choice != null -> sized
            else -> ""
        }
        return SendLine(text, null, "", err != null)
    }
    // ai: the rate the code carries, once sending
    override val figures get() = if (on && s.stats.offeredKBs > 0) Readout.rate(s.stats.offeredKBs) else ""
    override val heat get() = heatText(a.heat, a.clocks)
    override val railCapacity get() = if (on) s.capacityKBs else null
    override val settings: List<Setting> get() = listOfNotNull(
        // ai: what the symbol carries, then how this phone paints and shows it (one list since 2026-11-01; headed Code and
        // ai: Display from 2026-10-23)
        // ai: the slider sets the format's size (its version, sub-channels / 16); its title the blocks that size carries
        autoSlide("Blocks", s.blocks, s.picked / Pick.STEP, Pick.VERSION_MIN..Pick.VERSION_MAX, { n -> blocksText(Native.blocksFor(Pick.STEP * n)) }, { s.settle() }) { s.chooseBlocks(it) },
        if (s.vsyncOn) ChipChoice("FPS", s.rate, Pick.FPS.map { it to "$it" }) { s.chooseRate(it) }
        else Slide("FPS", "${s.rate}", s.rate.toFloat(), 1f..60f, 58, { s.settle() }) { v -> val r = v.roundToInt(); if (r != s.rate) s.chooseRate(r, now = false) },
        ChipChoice("Ring", s.ringIndex, Pick.RINGS.indices.map { it to "${Pick.RINGS[it]}" }) { s.chooseRing(it) },
        // ai: one code, or two side by side for a receiver's 2:1 crop (the web's #codes)
        ChipChoice("Codes", s.codes, listOf(1 to "One", 2 to "Two")) { s.chooseCodes(it) },
        // ai: with nothing chosen the setting stays auto and the painter running is the chip marked; a choice takes it outright
        ChipChoice("Encoder", if (s.painter == "auto") s.stats.painter.ifEmpty { "gpu" } else s.painter, listOf("gpu" to "GPU", "cpu" to "CPU")) { s.choosePainter(it) },
        ChipChoice("VSync", s.vsyncOn, listOf(true to "On", false to "Off")) { s.chooseVsync(it) },
        autoSlide("Brightness", s.brightness, 0, 1..100, { "$it%" }, {}, auto = "System") { if (it != s.brightness) s.chooseBrightness(it) },
        // ai: the space between two codes' rings, modules (the web's #spacing): the painter's gap and the two quiet zones
        if (s.codes == 2) Slide("Spacing", "${s.gap + SPACING_PAST} modules", (s.gap + SPACING_PAST).toFloat(), SPACING_PAST.toFloat()..(64f + SPACING_PAST), 63, { s.settle() }) { v ->
            val g = v.roundToInt() - SPACING_PAST; if (g != s.gap) s.chooseGap(g) } else null,
        // ai: the light round the code (SendState.margin), its label live and the box resized at the release
        Slide("Margin", "${marginDrag ?: s.margin} modules", (marginDrag ?: s.margin).toFloat(), Pick.QUIET.toFloat()..MARGIN_MAX.toFloat(), MARGIN_MAX - Pick.QUIET - 1,
            { marginDrag?.let { s.chooseMargin(it) }; marginDrag = null }) { v -> marginDrag = v.roundToInt() })
    private var marginDrag by mutableStateOf<Int?>(null)
    override val experimental: List<Setting> get() = emptyList()
    override val tools: List<Setting> get() = listOf(
        ChipChoice("Payload", s.test, listOf(false to "File", true to "Test stream"), enabled = s.phase != SendState.Phase.On && s.phase != SendState.Phase.Preparing) { s.test = it })
    override val readout get() = sendLab(s).takeIf { it.isNotEmpty() }?.let { Code(it) }
    override val about get() = a.aboutLine()
    // ai: While a send runs the code is the whole screen (2026-10-26): the code alone, the bars hidden (SendState.screenFor);
    // ai: the page only while paused; the collapser over it pauses and opens the page with the column
    override val codeAlone get() = on && !s.paused
    override val edgeTap = true
    override fun leaveCodeAlone() { s.pause(true); if (a.isOpen("sendCollapsed")) a.toggle("sendCollapsed") }
    override val codePadding = 16.dp
    // ai: the code's box, its shape the code's; while it holds the file upload and something is chosen, at least
    // ai: UPLOAD_ONE_ROW tall, so a 2:1 box no longer hides the selection under its tiles. In landscape, idle with the
    // ai: upload, the whole room (2026-10-31: the tiles stretched across it, at its top); from Start the code's shape,
    // ai: which the surface is made at
    override fun codeSize(maxWidth: Dp, maxHeight: Dp, portrait: Boolean): DpSize? {
        val ar = s.aspect.toFloat()
        val tall = uploading() && s.items.isNotEmpty()
        if (portrait) { val h = maxWidth / ar; return DpSize(maxWidth, if (tall) maxOf(h, UPLOAD_ONE_ROW) else h) }
        if (uploading() && !sending()) return null
        val w = minOf(maxWidth, maxHeight * ar)
        return DpSize(w, if (tall && !codeAlone) minOf(maxHeight, maxOf(w / ar, UPLOAD_ONE_ROW)) else w / ar)
    }
    override val rootModifier: Modifier = Modifier

    // ai: whether the code's box holds the file upload: no picture shown yet and a file the payload. A pick being walked
    // ai: or the bundle being made again (`reading`) keeps it: the state line says "Reading the files"
    private fun uploading() = !(sending() && s.shown) && !s.test
    private fun sending() = s.phase == SendState.Phase.On || s.phase == SendState.Phase.Preparing

    // ai: The code's box: while sending the code itself, painted into a SurfaceView the box's size (SendState.holder), the
    // ai: margin past its quiet zone the box's light round it; idle, the file upload; the test stream says so on the soft
    // ai: fill. The file's look stays over the surface until the first picture is presented (2026-10-07): a SurfaceView is
    // ai: black from its making until its first frame.
    @Composable
    override fun CodeArea(modifier: Modifier) {
        val sending = sending()
        val shown = sending && s.shown
        BoxWithConstraints(modifier.clipToBounds().background(if (shown || uploading()) Bg else Soft), contentAlignment = Alignment.Center) {
            if (sending) AndroidView({ ctx -> SurfaceView(ctx).apply { holder.addCallback(s.holder) } }, Modifier.fillMaxSize().padding(maxHeight * s.edgeShare.toFloat()))
            if (uploading()) Upload(listOf(Act("File", icon = Icons.file) { a.pickFile() }, Act("Folder", icon = Icons.folder) { a.pickFolder() },
                    Act("Media", icon = Icons.image) { a.pickMedia() }),
                s.items.map { UploadItem(it.key, it.name, if (it.dir) "${it.files} ${if (it.files == 1) "file" else "files"}" else Readout.bytes(it.size), it.dir) },
                enabled = !sending, landscape = maxWidth > maxHeight || LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE,
                onRemove = { s.remove(it) }, onClear = { s.clear() })
            else if (!shown) TestStreamNote()
        }
    }

    companion object {
        // ai: Spacing (2026-10-31, the desktop's and the web's name): the gap between two codes' rings plus their two quiet zones
        const val SPACING_PAST = 2 * Pick.QUIET
        private fun blocksText(b: Int) = "$b, ${"%.1f".format(Locale.ROOT, b * Pick.PAYLOAD / 1000.0)}\u00a0KB"
    }
}

// ai: Margin's bounds and default in modules, the web's (send.mjs marginOf) and the desktop's (Room.kt)
private const val MARGIN_MAX = 34
private const val MARGIN_DEFAULT = 8

// ai: The lab line (the web's #tx, its first lines): the format, where it is painted, the rate shown and a frame's paint,
// ai: and why not the GPU where auto fell back; for a file its own size and the bytes that go, and which pass shows;
// ai: empty before a first send.
private fun sendLab(s: SendState): String {
    val st = s.stats
    if (st.label.isEmpty()) return ""
    return "${st.label} on the ${if (st.painter == "gpu") "GPU" else "CPU"}, %.1f frames/s, paint %.1f ms a frame".format(Locale.ROOT, st.shownFps, st.paintMs) +
        (if (st.gpuRing) "\nposted ${st.posted}, held {${st.held}}, late ${st.late}, behind ${st.behind}, ${st.filled} of ${st.slots} buffers filled" else "") +
        (if (s.refused.isNotEmpty()) "\nthe window's queue: the vsync present refused (${s.refused})" else "") +
        (if (!s.test && s.size > 0) "\nfile ${s.size} B" + (if (st.sentBytes in 1 until s.size) ", ${st.sentBytes} B as sent (zstd)" else "") +
            (if (st.pass > 0) ", pass ${floor(st.pass).toInt()}" else "") else "") +
        if (st.painter == "cpu" && s.painter != "cpu" && st.gpuWhy.isNotEmpty()) "\nnot the GPU: ${st.gpuWhy}" else ""
}
