package dev.lizard.receiver

import dev.lizard.pick.Pick
import dev.lizard.readout.Readout
import android.content.Context
import android.content.res.Configuration
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.unit.Dp
import androidx.compose.foundation.horizontalScroll
import androidx.compose.material3.AlertDialog
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
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.displayCutout
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import kotlinx.coroutines.delay
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.runtime.LaunchedEffect
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
import androidx.activity.compose.BackHandler

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
    // ai: The transfer made at the pick (2026-10-19): the file copied (or the bundle's tar written) into the cache and
    // ai: every chunk compressed (txCreate) on a thread of its own once the choice is read, so the ready line says what
    // ai: goes, "<compressed> (<raw> raw)", and Start takes it at once (or waits for one still being made). compressed:
    // ai: its bytes as they go, 0 until made, kept through a run; a stop makes the next one while the app is in front.
    var compressed by mutableStateOf(0L)
    private var made = 0L                 // ai: the transfer made for the choice, not yet sending (0: none)
    private var madeFile: File? = null    // ai: the cache file it maps
    private var runFile: File? = null     // ai: the one the run's transfer maps
    private var making = 0                // ai: the make under way's number: a newer make or pick orphans it
    private var makingNow = false
    private var startWhenMade = false     // ai: Start pressed while the transfer was being made
    var test by mutableStateOf(false)
    // ai: Blocks a frame (2026-10-02), the web sender's
    // ai: slider (send.html #blocks and #subchAuto): 0 for auto, the largest the code's square holds (Pick.pick), else 2
    // ai: to 128 set by hand, a block 8 sub-channels, painted as set whatever the square (the web's too); kept, as the
    // ai: web's #subch is (a kept 1, one block a frame, which no format has since 2026-10-25, reads as auto). picked: the
    // ai: sub-channels the last configure took, which auto's label names.
    var blocks by mutableStateOf(prefs.getInt("sendBlocks", 0).let { if (it in 2..128) it else 0 })
    var picked by mutableStateOf(0)
    // ai: VSync (2026-10-23, on by default, the desktop sender's): the rate voted on both surfaces (the window's and the
    // ai: ring's, jni.cpp), so the panel moves to a mode the rate divides; off, no vote, the system's choice. Kept.
    var vsyncOn by mutableStateOf(prefs.getBoolean("sendVsync", true))
    // ai: Pictures a second (2026-10-02), the web's #fps: 60 unless set (the word states it; a rate the display's refresh
    // ai: does not divide holds pictures unevenly); kept, as the web's is. "Receiving with" (the app 60, a browser 24) went
    // ai: the same day. Under VSync Pick.FPS's menu (2026-10-23; a kept rate read as the nearest), without it a 1 to 60
    // ai: slider; VSync turned on takes a slider's rate to the nearest of the menu's.
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
    // ai: Margin (2026-10-27; the web's and the desktop's since 2026-10-23):
    // ai: the light between a code's ring and the box's edge, modules, its quiet zone (Pick.QUIET) at least and
    // ai: MARGIN_MAX at most, MARGIN_DEFAULT unless set; kept (sendMargin). Not painted: the box keeps it round the code's
    // ai: surface (CodeBox), as the desktop's presenter keeps it round the frame.
    var margin by mutableStateOf(prefs.getInt("sendMargin", MARGIN_DEFAULT).coerceIn(Pick.QUIET, MARGIN_MAX))
    private val sideModules get() = Pick.modules() + 2.0 * Pick.QUIET   // ai: a painted symbol's side, its quiet zone in
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
                make()
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
            if (depth > 64) throw java.io.IOException("$path is nested past 64 folders")
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
    fun chooseBlocks(b: Int) { val n = b.coerceIn(0, 128).let { if (it == 1) 0 else it }; if (n == blocks) return; blocks = n; prefs.edit().putInt("sendBlocks", blocks).apply(); unsettled = true }
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
    // ai: what the code offers at most, KB/s: blocks a frame x 469 B x pictures a second; null before the first configure
    val capacityKBs get() = if (picked > 0) codes * (if (stats.blocks > 0) stats.blocks else Native.blocksFor(picked)) * 469.0 * rate / 1000 else null
    fun choosePainter(p: String) { painter = p; prefs.edit().putString("sendPainter", p).apply(); if (phase == Phase.On) reconfigure() }
    fun chooseCodes(c: Int) { codes = c.coerceIn(1, 2); prefs.edit().putInt("sendCodes", codes).apply(); if (phase == Phase.On) reconfigure() }
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

    // ai: The first-send tip (2026-10-23; a tip, not a live warning: a power mode is no sure sign of dropped pictures, and
    // ai: modes differ from phone to phone): Start as a hand presses it (the
    // ai: column's, the rail's play) asks once, "Turn off power saving: it can make pictures drop.", with Cancel and
    // ai: Start; its Start sends and it is not asked again (the "lizard" preferences' sendTipSeen). The web's and the
    // ai: desktop's first-send tips say the same beside their brightness line; this app sets the brightness itself.
    var askTip by mutableStateOf(false)
    fun requestStart() { if (prefs.getBoolean("sendTipSeen", false)) start() else askTip = true }
    fun answerTip(go: Boolean) {
        askTip = false
        if (go) { prefs.edit().putBoolean("sendTipSeen", true).apply(); start() }
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
                if (why.isNotEmpty()) Log.i(Engine.TAG, "send: no GPU painter: $why")
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
        runFile?.delete(); runFile = null
        startWhenMade = false
        if (phase == Phase.On || phase == Phase.Preparing) phase = Phase.Idle
        shown = false; placed = false
        stats = Stats(); paused = false
        screenFor(false)
        // ai: the next Start's transfer, made afresh while the app is in front (a stop from onPause makes none: Start does)
        if (choice != null && made == 0L && !makingNow) a.window.decorView.post {
            if (a.lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.RESUMED) && phase !is Phase.Preparing && phase != Phase.On && choice != null && made == 0L && !makingNow) make()
        }
    }

    // ai: the format the code's square holds now (a turn of the phone re-picks it; the transfer goes on)
    private fun reconfigure() {
        room = roomOf()
        if (h == 0L || room <= 0) return
        val subch = if (blocks > 0) 8 * blocks else Pick.pick(room.toDouble())
        val threads = min(4, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val err = Native.txConfigure(h, subch, Pick.span(), fps, threads, when (painter) { "cpu" -> 0; "gpu" -> 1; else -> 2 }, assets, codes, gap, vsynced, vsyncOn)
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
        Log.i(Engine.TAG, "send: LIZARD-$subch${if (codes > 1) " x $codes, $gap modules apart" else ""}${if (blocks > 0) " set by hand" else ""} in a $room px square at $fps a second, painting on the " +
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
            if (p?.optBoolean("gpu") == true) Log.i(Engine.TAG, "send: posted ${stats.posted}, held {${held}}, late ${stats.late}, behind ${stats.behind}, ${stats.filled} of ${stats.slots} buffers filled, ${j.optInt("ahead")} of ${j.optInt("depth")} frames painted ahead")
            if (j.optString("error").isNotEmpty()) phase = Phase.Error(j.optString("error"))
        }
    }
    // ai: The vsync present refused for this send (jni.cpp txPost -1; its child surface let go): the frames go into the
    // ai: window's own queue from here, by the frame callback, the painters keeping host frames again (the configure
    // ai: with vsynced off). Why goes to the log and the lab line.
    private fun toWindowQueue(why: String) {
        if (!vsynced) return
        Log.i(Engine.TAG, "send: the vsync present refused (${why.ifEmpty { "no reason given" }}): frames into the window's queue")
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

// ai: Send, laid out as Receive since 2026-10-05 (a setup page, then a separate sending screen, until then): the code's
// ai: box where Receive has the camera, the code painted in it while sending and, idle, the file to send (a tap chooses
// ai: one, as the web sender's page); under it the state, the figures, the buttons and the rate, then Settings,
// ai: Developer Tools and About. Landscape: the column on the left and the code on the right, as large as the rest allows;
// ai: the column folds to a rail (play, pause or resume, the code's capacity, Home at its foot), as Receive's.
@Composable
internal fun MainActivity.SendScreen() {
    if (send.askTip) AlertDialog(onDismissRequest = { send.answerTip(false) },
        confirmButton = { Btn("Start", Kind.Primary) { send.answerTip(true) } },
        dismissButton = { Btn("Cancel", Kind.Text) { send.answerTip(false) } },
        text = { Text("Turn off power saving: it can make pictures drop.") },
        containerColor = Bg)
    val s = send
    val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    // ai: one code view for both layouts: a turn of the phone moves the surface rather than making another
    val code = remember { movableContentOf<Modifier> { m -> CodeBox(m) } }
    // ai: While a send runs the code is the whole screen (2026-10-26; a Fullscreen button did this from 2026-10-06 and is
    // ai: gone from the app): the code alone on the page's white, as large as the screen allows, the bars hidden
    // ai: (screenFor); Back stops the send and brings the page back. As the desktop's full screen: a tap within EDGE of the
    // ai: left edge shows the collapsed column over the code (Rail: its collapser, Pause or Resume, the rate, Home), a tap
    // ai: elsewhere or RAIL_MS untouched hides it. The page is left full screen only while paused (2026-10-27; the
    // ai: collapser opened it with the send running before):
    // ai: the collapser pauses and opens the page with the column, Resume takes the code full screen again. Preparing keeps
    // ai: the page (its Stop). The code's view moves between the two (its surface made again, SendState.holder).
    val running = s.phase == SendState.Phase.On
    if (running && !s.paused) {
        var rail by remember { mutableStateOf(false) }
        var touched by remember { mutableStateOf(0) }
        LaunchedEffect(rail, touched) { if (rail) { delay(RAIL_MS); rail = false } }
        BackHandler { s.stop() }
        // ai: the code and the rail clear of the display's cutout (2026-10-27: the app is edge to edge, and the S26's punch
        // ai: hole sat over the code's left margin), the page's white under it; the tap's edge measured from the screen's
        BoxWithConstraints(Modifier.fillMaxSize().background(Bg).pointerInput(Unit) {
            detectTapGestures { at -> rail = at.x < EDGE.toPx(); touched++ }
        }.windowInsetsPadding(WindowInsets.displayCutout), contentAlignment = Alignment.Center) {
            val a = s.aspect.toFloat()
            val w = minOf(maxWidth, maxHeight * a)
            code(Modifier.size(w, w / a))
            if (rail) Row(Modifier.align(Alignment.CenterStart).fillMaxHeight().background(Bg).pointerInput(Unit) { detectTapGestures { touched++ } }) {
                Rail(onClick = { rail = false; s.pause(true); if (isOpen("sendCollapsed")) toggle("sendCollapsed") },
                    foot = { Square { IconBtn(R.drawable.ic_back, "Home") { go(MainActivity.Screen.Home) } } }) { SendRail() }
                VerticalDivider(color = Line)
            }
        }
        return
    }
    val side: @Composable () -> Unit = {
        SendPanel()
        Fold("Settings", isOpen("sendSettings"), { toggle("sendSettings") }) {
            Fields {
                // ai: two groups (2026-10-23): Code, what the symbol carries, then Display, how this phone paints and shows it
                Group("Code") {
                    // ai: the slider sets the format's size (its sub-channels / 8); its title the blocks that size carries under the
                    // ai: format's rate profile and their bytes (2026-10-07)
                    AutoSlider("Blocks", s.blocks, s.picked / 8, 2..128, { n -> val b = Native.blocksFor(8 * n); "$b, ${"%.1f".format(Locale.ROOT, b * 469 / 1000.0)}\u00a0KB" }, { s.settle() }) { s.chooseBlocks(it) }
                    if (s.vsyncOn) Select("FPS", s.rate, Pick.FPS.map { it to "$it" }) { s.chooseRate(it) }
                    else Field("FPS", "${s.rate}") { Bar(s.rate.toFloat(), 1f..60f, 58, { s.settle() }) { v -> val r = v.roundToInt(); if (r != s.rate) s.chooseRate(r, now = false) } }
                    // ai: one code, or two side by side for a receiver's 2:1 crop (the web's #codes)
                    Select("Codes", s.codes, listOf(1 to "One", 2 to "Two")) { s.chooseCodes(it) }
                }
                Group("Display") {
                    // ai: the fields in the web's shape, a menu each (2026-10-06; chips until then); with nothing chosen the setting
                    // ai: stays auto and the menu shows the painter running; a choice takes that painter outright
                    Select("Encoder", if (s.painter == "auto") s.stats.painter.ifEmpty { "gpu" } else s.painter, listOf("gpu" to "GPU", "cpu" to "CPU")) { s.choosePainter(it) }
                    Select("VSync", s.vsyncOn, listOf(true to "On", false to "Off")) { s.chooseVsync(it) }
                    AutoSlider("Brightness", s.brightness, 0, 1..100, { "$it%" }, {}, auto = "System") { if (it != s.brightness) s.chooseBrightness(it) }
                    // ai: the gap between two codes (the web's #gap)
                    if (s.codes == 2) Field("Gap", "${s.gap} modules") { Bar(s.gap.toFloat(), 0f..64f, 63, { s.settle() }) { v -> val g = v.roundToInt(); if (g != s.gap) s.chooseGap(g) } }
                    // ai: the light round the code (SendState.margin), its label live and the box resized at the release
                    var m by remember { mutableStateOf<Int?>(null) }
                    Field("Margin", "${m ?: s.margin} modules") {
                        Bar((m ?: s.margin).toFloat(), Pick.QUIET.toFloat()..MARGIN_MAX.toFloat(), MARGIN_MAX - Pick.QUIET - 1, { m?.let { s.chooseMargin(it) }; m = null }) { v -> m = v.roundToInt() }
                    }
                }
            }
        }
        Fold("Developer Tools", isOpen("sendAdvanced"), { toggle("sendAdvanced") }) {
            Fields {
                Select("Payload", s.test, listOf(false to "File", true to "Test stream"), enabled = s.phase != SendState.Phase.On && s.phase != SendState.Phase.Preparing) { s.test = it }
                sendLab(s).let { if (it.isNotEmpty()) CodeBlock(it) }
            }
        }
        About()
        Spacer(Modifier.height(24.dp))
    }
    if (!landscape) Column(Modifier.fillMaxSize().background(Bg).safeDrawingPadding().padding(horizontal = 16.dp)) {
        TopBar("Send", onBack = { go(MainActivity.Screen.Home) }) { SwapBtn("Receiver") { go(MainActivity.Screen.Receive) } }
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            Spacer(Modifier.height(8.dp))
            // ai: the code's box, its shape the code's; while it holds the file upload and something is chosen, at least
            // ai: UPLOAD_ONE_ROW tall (the tiles and the card with a row of items, which scroll inside the card), so a 2:1
            // ai: box no longer hides the selection under its tiles
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                val h = maxWidth / s.aspect.toFloat()
                code(Modifier.fillMaxWidth().height(if (uploading() && s.items.isNotEmpty()) maxOf(h, UPLOAD_ONE_ROW) else h))
            }
            side()
        }
    } else BoxWithConstraints(Modifier.fillMaxSize().background(Bg).safeDrawingPadding()) {
        // ai: no narrower than Resume beside Stop with no label folded (2026-10-06, Fullscreen among them until 2026-10-26)
        val sideLo = CellsFloor(listOf("Start", "Stop", "Resume"), 2)
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
                // ai: square for one code, about 2:1 for two; while it holds the file upload with something chosen, as tall
                // ai: as the tiles and a row of items need, up to the room there is
                code(Modifier.size(w, if (uploading() && s.items.isNotEmpty()) minOf(maxHeight, maxOf(w / a, UPLOAD_ONE_ROW)) else w / a))
            }
        }
    }
}

// ai: full screen's rail (the Send page's code alone while a send runs): how near the left edge a tap shows it, and how
// ai: long it stays untouched
private val EDGE = 72.dp
private const val RAIL_MS = 4000L

// ai: the upload's height with one row of items: its padding, the tiles, the gap, the card's head and one row
// ai: Margin's bounds and default in modules, the web's (send.mjs marginOf) and the desktop's (Room.kt)
private const val MARGIN_MAX = 34
private const val MARGIN_DEFAULT = 8
private val UPLOAD_ONE_ROW = (16 + TILE + 16 + 4 + 48 + 4 + TILE + 12 + 16).dp

// ai: whether the code's box holds the file upload: no picture shown yet and a file the payload. A pick being walked
// ai: or the bundle being made again (`reading`, which every add, remove and Clear sets) keeps it: the state line says
// ai: "Reading the files" (the box swapped to a cover for that until 2026-10-26, a flash at every removal).
private fun MainActivity.uploading(): Boolean {
    val s = send
    val sending = s.phase == SendState.Phase.On || s.phase == SendState.Phase.Preparing
    return !(sending && s.shown) && !s.test
}

// ai: The code's box: while sending the code itself, painted into a SurfaceView the box's size (SendState.holder); idle,
// ai: the file upload (Upload).
@Composable
private fun MainActivity.CodeBox(modifier: Modifier) {
    val s = send
    val sending = s.phase == SendState.Phase.On || s.phase == SendState.Phase.Preparing
    // ai: the file's look stays over the surface until the first picture is presented (2026-10-07): a SurfaceView is
    // ai: black from its making until its first frame, through Preparing (a file compressed) and the first presents,
    // ai: and a box drawn over it in the window hides it
    val shown = sending && s.shown
    BoxWithConstraints(modifier.clipToBounds().background(if (shown || uploading()) Bg else Soft), contentAlignment = Alignment.Center) {
        // ai: the surface the frame's shape, the margin past its quiet zone the box's light round it (SendState.margin)
        if (sending) AndroidView({ ctx -> SurfaceView(ctx).apply { holder.addCallback(s.holder) } }, Modifier.fillMaxSize().padding(maxHeight * s.edgeShare.toFloat()))
        // ai: idle: the file upload (Upload) until the first picture shows; the test stream says so on the soft fill instead
        if (uploading()) Upload(sending)
        else if (!shown) Column(Modifier.fillMaxSize().background(Soft), verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(painterResource(R.drawable.ic_file), contentDescription = null, Modifier.size(24.dp), tint = Muted)
            Text("Test stream", style = MaterialTheme.typography.bodyMedium, color = Muted,
                textAlign = TextAlign.Center, modifier = Modifier.padding(horizontal = 16.dp))
        }
    }
}

// ai: The file upload (2026-10-26, LocalSend's geometry; the web's #pick): the File, Folder and Media tiles (Parts.kt
// ai: Tile), and under them, while the selection holds something, its card: the count with Clear (the state line says
// ai: the size), then a tile an item (ItemTile, its x removing it), the items in rows stretched across the column
// ai: (TileRows, 104 dp the least before a fold), the three tiles always one row; nothing changes while sending. Centred
// ai: in the box; only the card scrolls: its items one row scrolling sideways in landscape, rows scrolling down in portrait.
@Composable
private fun MainActivity.Upload(sending: Boolean) {
    val s = send
    val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    BoxWithConstraints(Modifier.fillMaxSize().background(Bg)) {
        val w = minOf(maxWidth - 32.dp, 576.dp)
        Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically), horizontalAlignment = Alignment.CenterHorizontally) {
            val tiles = listOf(Triple(R.drawable.ic_file, "File") { pickFile() }, Triple(R.drawable.ic_folder, "Folder") { pickFolder() },
                Triple(R.drawable.ic_image, "Media") { pickMedia() })
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                tiles.forEach { (icon, label, act) -> Tile(icon, label, (w - 12.dp * (tiles.size - 1)) / tiles.size, enabled = !sending, onClick = act) }
            }
            if (s.items.isNotEmpty()) Column(Modifier.width(w).weight(1f, fill = false).clip(RoundedCornerShape(12.dp)).background(Card)
                .padding(start = 12.dp, end = 12.dp, top = 4.dp, bottom = 12.dp)) {
                Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).padding(start = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("${s.items.size} ${if (s.items.size == 1) "item" else "items"}", style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
                    Btn("Clear", Kind.Text, enabled = !sending) { s.clear() }
                }
                val items = s.items.toList()
                val item: @Composable (Int, Dp) -> Unit = { i, iw ->
                    val it = items[i]
                    ItemTile(it.name, if (it.dir) "${it.files} ${if (it.files == 1) "file" else "files"}" else Readout.bytes(it.size),
                        if (it.dir) R.drawable.ic_folder else R.drawable.ic_file, iw, enabled = !sending) { s.remove(it.key) }
                }
                if (landscape) {
                    // ai: one row at the columns' width (TileRows' rule), scrolling sideways past what fits
                    val gap = 8.dp; val inner = w - 24.dp
                    val cols = maxOf(1, ((inner + gap) / (TILE.dp + gap)).toInt())
                    Row(Modifier.padding(top = 4.dp).horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(gap)) {
                        items.indices.forEach { item(it, (inner - gap * (cols - 1)) / cols) }
                    }
                } else Box(Modifier.padding(top = 4.dp).weight(1f, fill = false).verticalScroll(rememberScrollState())) { TileRows(items.size, w - 24.dp, 8.dp, item) }
            }
        }
    }
}

// ai: The state, the buttons and the rate, as Receive's transfer panel: the state line (the file and its size,
// ai: "Preparing", "Sending" with the bytes that go, "Paused", or what went wrong), then Start, or Stop (Resume beside it
// ai: while paused), then the rate the code carries.
@Composable
private fun MainActivity.SendPanel() {
    val s = send
    val on = s.phase == SendState.Phase.On
    val busy = on || s.phase == SendState.Phase.Preparing
    val st = s.stats
    val what = if (s.test) "the test stream" else s.name
    // ai: what goes, "<compressed> (<raw> raw)" once the transfer is made, "<raw> raw" while it is being made (2026-10-19,
    // ai: the web's and the desktop's the same); the pass is Developer Tools' (sendLab, 2026-10-05)
    val count = s.choice?.takeIf { it.single == null }?.files?.let { "$it ${if (it == 1) "file" else "files"}, " } ?: ""
    // ai: "raw" kept with its figure (no-break spaces, 2026-10-23: "(47 KB / raw)" broke there)
    val sizeText = if (s.compressed > 0) "${Readout.bytes(s.compressed)} (${Readout.bytes(s.size)}\u00a0raw)" else "${Readout.bytes(s.size)}\u00a0raw"
    val sized = if (s.test || s.size <= 0) what else "$what, ${if (on) "" else count}$sizeText"
    val err = (s.phase as? SendState.Phase.Error)?.why
    val line = when {
        err != null -> err
        s.phase == SendState.Phase.Preparing -> "Preparing $what"
        on && s.paused -> "Paused"
        on -> "Sending $sized"
        s.test -> "The test stream"
        s.reading -> "Reading the files"
        s.choice != null -> sized
        else -> ""
    }
    // ai: the web sender's order (2026-10-06): the buttons first (Start; Stop while preparing; paused, Resume beside Stop),
    // ai: then the state line, then the rate as the web's figures line, 14 sp grey. No Fullscreen (2026-10-26): a send
    // ai: running is the whole screen.
    Column(Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 4.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (!busy) Btn("Start", Kind.Primary, enabled = s.test || s.ready, modifier = Modifier.fillMaxWidth()) { s.requestStart() }
        else if (s.paused) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Btn("Resume", Kind.Primary, modifier = Modifier.weight(1f)) { s.pause(false) }
            Btn("Stop", modifier = Modifier.weight(1f)) { s.stop() }
        } else Btn("Stop", modifier = Modifier.fillMaxWidth()) { s.stop() }
    }
    // ai: a line naming the file holds its head on one line (Parts.kt NameLine, 2026-10-23)
    val named = !s.test && err == null && (s.phase == SendState.Phase.Preparing || (on && !s.paused) || (!on && !s.reading && s.choice != null))
    if (named) NameLine(when { s.phase == SendState.Phase.Preparing -> "Preparing "; on -> "Sending "; else -> "" }, what,
        if (s.phase == SendState.Phase.Preparing || s.size <= 0) "" else ", ${if (on) "" else count}$sizeText",
        MaterialTheme.typography.titleMedium, Fg, Modifier.padding(top = 4.dp, bottom = 2.dp))
    else if (line.isNotEmpty()) Text(line, style = MaterialTheme.typography.titleMedium, color = if (err != null) Bad else Fg, modifier = Modifier.padding(top = 4.dp, bottom = 2.dp))
    if (on && st.offeredKBs > 0) Text(Readout.rate(st.offeredKBs), style = MaterialTheme.typography.bodyMedium, color = Muted, modifier = Modifier.padding(top = 2.dp))
    HeatWarning(heat, clocks, modifier = Modifier.padding(top = 8.dp))
    Spacer(Modifier.height(8.dp))
}

// ai: The rail's squares: play to start (or to resume), pause while sending, and the code's capacity while it is shown;
// ai: above the play or pause, the heat warning's red triangle while the phone throttles (2026-10-08, as Receive's)
@Composable
private fun MainActivity.SendRail() {
    val s = send
    val on = s.phase == SendState.Phase.On
    heatText(heat, clocks)?.let { HeatSquare(it) }
    Square {
        IconBtn(if (on && !s.paused) R.drawable.ic_pause else R.drawable.ic_play, if (!on) "Start" else if (s.paused) "Resume" else "Pause",
            enabled = if (on) true else s.phase != SendState.Phase.Preparing && (s.test || s.ready)) { if (on) s.pause(!s.paused) else s.requestStart() }
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
        (if (s.refused.isNotEmpty()) "\nthe window's queue: the vsync present refused (${s.refused})" else "") +
        (if (!s.test && s.size > 0) "\nfile ${s.size} B" + (if (st.sentBytes in 1 until s.size) ", ${st.sentBytes} B as sent (zstd)" else "") +
            (if (st.pass > 0) ", pass ${floor(st.pass).toInt()}" else "") else "") +
        if (st.painter == "cpu" && s.painter != "cpu" && st.gpuWhy.isNotEmpty()) "\nnot the GPU: ${st.gpuWhy}" else ""
}

// ai: Blocks a frame (send.html #blocks and #subchAuto), 0 for auto: the slider's leftmost step is auto (2026-10-05; an
// ai: Auto chip at the title row's end until then), its value "Auto, 60, 28.1 KB" while what auto takes is known
// ai: ("Auto" before the first configure), every step right of it a count by hand. value the setting, shown what auto
// ai: takes now (0 while unknown), range the counts; the step left of them auto. Brightness the same with its leftmost
// ai: step the phone's own (`auto` "System", 2026-10-27).
@Composable
private fun AutoSlider(title: String, value: Int, shown: Int, range: IntRange, text: (Int) -> String, done: () -> Unit, auto: String = "Auto", choose: (Int) -> Unit) {
    val isAuto = value == 0
    val b = if (isAuto) shown else value
    val lo = range.first - 1
    Field(title, if (isAuto) (if (b > 0) "$auto, ${text(b)}" else auto) else text(b)) {
        Bar((if (isAuto) lo else value).toFloat(), lo.toFloat()..range.last.toFloat(), range.last - lo - 1, done) { val v = it.roundToInt(); choose(if (v < range.first) 0 else v) }
    }
}
