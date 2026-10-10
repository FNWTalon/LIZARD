package dev.lizard.receiver

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ContentValues
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.pm.ShortcutInfo
import android.content.pm.ShortcutManager
import android.graphics.drawable.Icon
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.provider.MediaStore
import android.provider.Settings as AndroidSettings
import android.util.Log
import android.view.WindowManager
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.Lifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.math.abs
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

// ai: The app (2026-10-01): three screens, Home (Send and Receive, the received files, and at the bottom Settings: the
// ai: tips again, the received files' size, About), Receive (the camera and the transfer, then Settings and Developer
// ai: Tools, rows that open, and About) and Send (a
// ai: file, Start, Settings, Developer Tools, About); since 2026-10-02 as the web's pages after their tidying, the Settings
// ai: screen and its gears gone. One activity, its `screen` and the system back between them, no navigation library.
// ai: The web's pages are the same shell (lizard-web/index.html, recv.html, send.html) and the words the
// ai: same (Readout.kt). The camera runs on Receive alone, from entering it (or a resume there) to leaving it (or a
// ai: pause, or Stop camera); the receiver, and a transfer partly in, outlives it as it outlives Stop. The launcher opens
// ai: Home; `--es screen receive` (tools/phone/ab.sh) and the launcher shortcut (publishShortcut) open Receive with the
// ai: camera, in this activity when it runs (singleTop, onNewIntent).
class MainActivity : ComponentActivity() {
    // ai: the folds' keys, and the two collapsers' (Receive's column, the sending screen's panel; landscape)
    companion object {
        val FOLDS = listOf("recvSettings", "recvAdvanced", "sendSettings", "sendAdvanced", "recvCollapsed", "sendCollapsed")
        // ai: Save replays: the frames a replay keeps, and the seconds of stats rows its stats.jsonl holds (the web's
        // ai: REC_FRAMES and REC_LOG, recv.mjs)
        const val REPLAY_FRAMES = 300
        const val REPLAY_LOG = 60
        // ai: The replays are the process's, not an activity's (2026-10-03: each activity kept its own, so
        // ai: one made again while a run ended adopted an older run, which the old one's end then deleted, and never
        // ai: listed the run that ended). `runs` in the order begun (main thread only): the newest finished run, the
        // ai: runs not yet ended, and a failed one until the next begins. `busyRuns` the runs being ended or packed,
        // ai: whose folders nothing else deletes; `exporting` the run a Download is packing, `replayNote` how the last
        // ai: Download went; `adopted` once the runs on disk were read, at the process's first activity.
        internal val runs = mutableStateListOf<Replay>()
        val busyRuns: MutableSet<String> = java.util.Collections.synchronizedSet(HashSet())
        internal var exporting by mutableStateOf("")
        internal var replayNote by mutableStateOf("")
        private var adopted = false
    }
    enum class Screen { Home, Receive, Send }

    internal lateinit var engine: Engine
    internal lateinit var library: Library
    private lateinit var prefs: SharedPreferences
    internal var phase by mutableStateOf<Engine.Phase>(Engine.Phase.Idle)
    internal var cam by mutableStateOf<Engine.CamInfo?>(null)
    internal var settings by mutableStateOf(Settings())
    internal var screen by mutableStateOf(Screen.Home)
    internal var stopped by mutableStateOf(false)          // ai: Receive: Stop camera pressed, the camera off until Start
    internal var granted by mutableStateOf(false)
    internal var denied by mutableStateOf(false)           // ai: refused, and Android asks no more: only its settings allow it
    internal var note by mutableStateOf("")                // ai: what came of a Save a copy or an Open
    internal var files by mutableStateOf(listOf<Library.Entry>())
    internal var tipsSeen by mutableStateOf(false)
    // ai: The landscape column's width on Receive and Send, dp (2026-10-05): 0 the default (SideWidth), else what its edge
    // ai: was dragged to (SideEdge), kept as `sideDp`; sideDragging while a finger is on the edge (Send re-picks after).
    internal var sideDp by mutableStateOf(0f)
    internal var sideDragging by mutableStateOf(false)
    internal fun saveSide() { prefs.edit().putFloat("sideDp", sideDp).apply() }
    // ai: the rows that open, open or not, kept by key (the web's send:dev, send:logs, recv:dev, recv:logs)
    internal val folds = mutableStateMapOf<String, Boolean>()
    internal var autoRan by mutableStateOf("")             // ai: what auto last decoded on, GPU or CPU (the Decoder menu's value under auto)
    internal var raw by mutableStateOf("")                 // ai: the stats JSON as it came, for Developer Tools
    internal var rx by mutableStateOf(Readout.Rx())
    internal var root by mutableStateOf("")                // ai: the transfer in hand's BLAKE3 root, hex
    private var filing = ""      // ai: the root whose keep is under way (Engine.keep), until its answer
    private var unfiled = ""     // ai: a root whose keep failed: not tried again in this run of the app
    private var ended = ""       // ai: the root whose arrival stopped the camera (once a root)
    private var saving: Library.Entry? = null
    private var asked = false
    // ai: Save replays (Developer Tools, 2026-10-03; the web's Record 300 frames, recv.mjs saveRun, as a rolling
    // ai: window): while the switch is on and the camera runs, the newest REPLAY_FRAMES frames the decoder is handed
    // ai: are kept as a run's folder under filesDir/replays (liblizard/core/rx/replay.h), its stats.jsonl the last
    // ai: REPLAY_LOG stats rows, made a second while a run records and at no other time (2026-10-09: rows are logged only
    // ai: as part of a replay; the rig's dev log went); the run ends when either stops. A finished run keeps its Download
    // ai: until a newer run ends with frames, then goes (the companion's `runs`). `owner` the Engine recording the run,
    // ai: which this activity's poll alone ends; null once it ends.
    enum class ReplayState { Recording, Ending, Ready, Failed }
    internal data class Replay(val run: String, val dir: File, val handle: Long, val state: ReplayState, val frames: Long = 0, val bytes: Long = 0,
                               val why: String = "", val owner: Any? = null)
    private val statsRows = ArrayDeque<String>()
    private fun nz(x: Double): Any = if (x.isNaN() || x.isInfinite()) JSONObject.NULL else x   // ai: a JSON number, or null where there is none
    private val main = Handler(Looper.getMainLooper())   // ai: a Download's end, which may come after this activity's

    private val ask = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        granted = ok
        denied = !ok && !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)
        if (ok && wantsCamera() && resumed()) startCamera()
    }
    // ai: the app's sender (Send.kt) and the file chooser it opens
    internal val send by lazy { SendState(this) }
    private val choose = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> if (uri != null) send.pick(uri) }
    internal fun pickFile() { try { choose.launch(arrayOf("*/*")) } catch (_: ActivityNotFoundException) {} }
    private val save = registerForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        val e = saving
        saving = null
        if (uri != null && e != null) saveTo(uri, e)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // ai: dark status-bar icons on the white screen whatever the system's theme
        enableEdgeToEdge(SystemBarStyle.light(android.graphics.Color.TRANSPARENT, android.graphics.Color.TRANSPARENT),
            SystemBarStyle.light(android.graphics.Color.TRANSPARENT, android.graphics.Color.TRANSPARENT))
        // ai: a debuggable build shows over the lock screen and wakes it, so a run driven over adb reaches the camera
        // ai: on a locked phone; a release build does neither
        if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) { setShowWhenLocked(true); setTurnScreenOn(true) }
        prefs = getSharedPreferences(Settings.PREFS, MODE_PRIVATE)
        sideDp = prefs.getFloat("sideDp", 0f)
        tipsSeen = prefs.getBoolean("tipsSeen", false)
        for (k in FOLDS) folds[k] = prefs.getBoolean(k, false)
        library = Library(File(filesDir, "library"))
        adoptReplays()
        files = library.list()
        engine = Engine(applicationContext, { p ->
            phase = p
            // ai: the screen stays on while the camera runs
            if (p == Engine.Phase.On) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            else if (p !is Engine.Phase.Starting && p !is Engine.Phase.Loading) window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            // ai: and full screen, the system bars hidden, from the camera's start until it stops (2026-10-05, as Send does
            // ai: while sending: SendState.screenFor); leaving Receive stops the camera, which brings them back
            val bars = WindowCompat.getInsetsController(window, window.decorView)
            if (p == Engine.Phase.On || p is Engine.Phase.Starting || p is Engine.Phase.Loading) bars.hide(WindowInsetsCompat.Type.systemBars())
            else bars.show(WindowInsetsCompat.Type.systemBars())
        }, { cam = it })
        settings = Settings.load(this) { camId(it) }   // ai: after the engine: a lens's own settings are under the id it resolves
        granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        // ai: a restored activity (a configuration change, the process gone and back) keeps the screen it was on; only a
        // ai: fresh one takes its launch intent's (2026-10-01: restored, it went back to the intent's Receive)
        savedInstanceState?.getString("screen")?.let { s -> Screen.entries.firstOrNull { it.name == s }?.let { screen = it } }
        if (savedInstanceState == null) handle(intent)
        publishShortcut()
        setContent { MaterialTheme(colorScheme = LizardColours, typography = LizardType) { App() } }
    }

    override fun onNewIntent(intent: Intent) { super.onNewIntent(intent); handle(intent) }
    // ai: Receive from the shortcut or ab.sh; Send from its shortcut, or with a file another app shared to LIZARD
    private fun handle(i: Intent?) {
        if (i?.action == Intent.ACTION_SEND) {
            @Suppress("DEPRECATION") val u = i.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
            if (u != null) send.pick(u)
            go(Screen.Send)
            return
        }
        when (i?.getStringExtra("screen")) { "receive" -> go(Screen.Receive); "send" -> go(Screen.Send) }
    }
    // ai: The launcher icon's long-press shortcut, Receive: dynamic, so its intent carries the flags given here. A static
    // ai: one (res/xml/shortcuts.xml until 2026-10-01) is always started with FLAG_ACTIVITY_CLEAR_TASK, which finished
    // ai: this activity, its engine and a transfer partly in, and made another beside it. From here the
    // ai: launcher's new-task flag finds this task, CLEAR_TOP lets go of anything above this activity, and SINGLE_TOP
    // ai: hands the intent to onNewIntent. Published at every start, which a foreground app may do at no cost.
    private fun publishShortcut() = runCatching {
        val i = Intent(Intent.ACTION_VIEW, null, this, MainActivity::class.java).putExtra("screen", "receive")
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        val sendI = Intent(Intent.ACTION_VIEW, null, this, MainActivity::class.java).putExtra("screen", "send")
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        getSystemService(ShortcutManager::class.java)?.dynamicShortcuts = listOf(
            ShortcutInfo.Builder(this, "receive-camera").setShortLabel(getString(R.string.receive)).setIcon(Icon.createWithResource(this, R.drawable.ic_camera)).setIntent(i).build(),
            ShortcutInfo.Builder(this, "send-file").setShortLabel(getString(R.string.send)).setIcon(Icon.createWithResource(this, R.drawable.ic_send)).setIntent(sendI).build())
    }.onFailure { Log.w(Engine.TAG, "shortcut: ${it.message}") }
    override fun onSaveInstanceState(outState: Bundle) { super.onSaveInstanceState(outState); outState.putString("screen", screen.name) }
    // ai: The phone's heat (2026-10-01): Android's thermal status, 0 (none) to 6 (shutdown), followed while the
    // ai: activity is in front (the listener is told the status at once on registering). Receive and Send warn from
    // ai: moderate (2) up: the S26's camera fell from 56 to 43 frames a second at 4K there (STATUS "The default ring
    // ai: the 128, and the 4K camera").
    internal var heat by mutableStateOf(0)
    // ai: the GPU's clock ceiling against its top (Clocks.kt), read once a second by the poll; null where unreadable
    internal var clocks by mutableStateOf<Clocks.Read?>(null)
    // ai: The whole phone's power off the charger (2026-10-08, for the heat's levers): the battery's discharge current
    // ai: (BatteryManager CURRENT_NOW, uA, negative while it discharges on the S26) times its voltage, null while
    // ai: plugged in, when it measures the charger instead; read once a second, the stats row's `powerW` and
    // ai: `batteryC`, a `power:` log line every 5 s with the mean.
    internal var powerW: Double? = null
    internal var batteryC = 0.0
    private var powerSum = 0.0
    private var powerN = 0
    private fun readPower() {
        val b = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return
        batteryC = b.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10.0
        val plugged = b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0
        val ua = getSystemService(BatteryManager::class.java)?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) ?: Int.MIN_VALUE
        val mv = b.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0)
        powerW = if (plugged || ua == Int.MIN_VALUE || mv <= 0) null else abs(ua.toDouble()) * mv / 1e9
        powerW?.let { powerSum += it; powerN++ }
    }
    // ai: `adb shell setprop debug.lizard.dim <0..1>` (2026-10-08, a lab switch for the heat's levers): Receive's
    // ai: window brightness while the camera runs, acted on when the value changes; anything else, the system's
    private var dimAsked = ""
    private val heatListener = PowerManager.OnThermalStatusChangedListener { if (it != heat) Log.i(Engine.TAG, "thermal status $it"); heat = it }
    override fun onResume() {
        super.onResume()
        Log.i(Engine.TAG, "resume")
        getSystemService(PowerManager::class.java)?.addThermalStatusListener(mainExecutor, heatListener)
        // ai: the permission again: it may have been given in Android's settings meanwhile
        granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (granted) denied = false
        if (wantsCamera()) startCamera()
        else if (screen == Screen.Receive && !granted && !asked) askCamera()
    }
    override fun onPause() {
        super.onPause(); Log.i(Engine.TAG, "pause"); engine.stop(); send.stop()
        getSystemService(PowerManager::class.java)?.removeThermalStatusListener(heatListener)
    }
    override fun onDestroy() { super.onDestroy(); runs.filter { it.state == ReplayState.Recording && it.owner === engine }.forEach { endReplay(it) }; engine.shutdown() }

    private fun resumed() = lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)
    private fun wantsCamera() = screen == Screen.Receive && !stopped && granted
    internal fun startCamera() { phase = Engine.Phase.Starting; engine.start(settings) }

    // ai: Between screens. Leaving Receive stops the camera; entering it starts it (only while resumed: onResume starts it
    // ai: otherwise, so a launch into Receive opens the camera once), asking for the permission the first time.
    internal fun go(s: Screen) {
        if (s == screen) return
        if (screen == Screen.Receive) engine.stop()
        if (screen == Screen.Send) send.stop()
        screen = s
        note = ""
        if (s == Screen.Home) files = library.list()
        if (s == Screen.Receive) {
            stopped = false
            if (resumed()) { if (granted) startCamera() else askCamera() }
        }
    }
    internal fun toggleCamera() {
        if (phase == Engine.Phase.On || phase == Engine.Phase.Starting || phase == Engine.Phase.Loading) { stopped = true; engine.stop() }
        else { stopped = false; startCamera() }
    }
    internal fun askCamera() { asked = true; ask.launch(Manifest.permission.CAMERA) }
    internal fun openAppSettings() {
        try { startActivity(Intent(AndroidSettings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null))) } catch (_: ActivityNotFoundException) {}
    }

    // ai: the back camera the camera switch opens (Engine.rearId), the id the lens's own settings are kept under
    // ai: (Settings.forCamera, 2026-10-04); null where the phone has none or the camera service fails
    private fun camId(camera: String): String? = runCatching { engine.rearId(camera) }.getOrNull()
    // ai: A Settings or Developer Tools switch: kept, and the camera started again where the switch needs it. Another lens
    // ai: chosen brings its own resolution and zoom (as last kept for it, else the defaults).
    internal fun change(s0: Settings, restart: Boolean) {
        val cam = camId(s0.camera)
        val s = if (s0.camera != settings.camera) s0.forCamera(this, cam) else s0
        settings = s
        s.save(this, cam)
        engine.batch(s.frames)
        if (restart && screen == Screen.Receive && phase != Engine.Phase.Idle && granted) startCamera()
    }
    // ai: Settings' zoom slider: the camera moved live (Engine.zoom, no restart), the setting kept at every step
    internal fun zoomTo(z: Float) {
        if (settings.zoom.toFloatOrNull() == z) return
        settings = settings.copy(zoom = "%.1f".format(java.util.Locale.ROOT, z))
        engine.zoom(z)
        settings.save(this, camId(settings.camera))
    }
    internal fun tipsDone() { tipsSeen = true; prefs.edit().putBoolean("tipsSeen", true).apply() }
    internal fun tipsAgain() { tipsSeen = false; prefs.edit().putBoolean("tipsSeen", false).apply() }
    internal fun isOpen(k: String) = folds[k] == true
    internal fun toggle(k: String) { val v = !isOpen(k); folds[k] = v; prefs.edit().putBoolean(k, v).apply() }

    // ai: Receive again (Clear until 2026-10-01) went 2026-10-05: a second Start camera; a receiver tests with the test stream.

    // ai: The received file's actions, through the FileProvider (AndroidManifest.xml), as the header's media type.
    private fun uriOf(e: Library.Entry): Uri = FileProvider.getUriForFile(this, "$packageName.files", e.file)
    internal fun open(e: Library.Entry) {
        val i = Intent(Intent.ACTION_VIEW).setDataAndType(uriOf(e), e.mime).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        try { startActivity(i) } catch (_: ActivityNotFoundException) { note = "No app on this phone opens ${e.mime} files: share it, or save a copy." }
    }
    internal fun share(e: Library.Entry) {
        val uri = uriOf(e)
        val i = Intent(Intent.ACTION_SEND).setType(e.mime).putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        i.clipData = ClipData.newRawUri(e.name, uri)
        try { startActivity(Intent.createChooser(i, null)) } catch (_: ActivityNotFoundException) {}
    }
    internal fun saveCopy(e: Library.Entry) { saving = e; note = ""; save.launch(e.name) }   // ai: the Save buttons (a copy where the user picks)
    private fun saveTo(uri: Uri, e: Library.Entry) {
        Thread {
            val n = try {
                contentResolver.openOutputStream(uri)!!.use { o -> e.file.inputStream().use { it.copyTo(o, 1 shl 16) } }
                "Saved ${e.name}"
            } catch (x: Exception) { "The copy could not be saved: ${x.message}" }
            runOnUiThread { note = n }
        }.start()
    }
    // ai: deleting the file the receiver shows as received forgets it there too (2026-10-07): the state line's green
    // ai: "Received" clears at the next poll, and the file in the light is received anew
    internal fun delete(e: Library.Entry) { if (e.root.isNotEmpty() && e.root == root) forgetReceived(); library.delete(e); files = library.list() }
    internal fun deleteAll() { if (files.any { it.root.isNotEmpty() && it.root == root }) forgetReceived(); library.deleteAll(); files = library.list() }
    private fun forgetReceived() { engine.clear(); root = ""; unfiled = "" }
    internal fun installed(): String = try {
        SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.ROOT).format(Date(packageManager.getPackageInfo(packageName, 0).lastUpdateTime))
    } catch (_: Exception) { "" }
    internal fun version(): String = try { packageManager.getPackageInfo(packageName, 0).versionName ?: "" } catch (_: Exception) { "" }

    // ai: A verified file into the received files, once a root: moved off the native store by the engine (Engine.keep), so
    // ai: a receiver's rebuild no longer takes it. A root already kept is not moved again, and none is begun while one
    // ai: is under way: Library.dest empties the root's folder, which a second keep of the same root (a poll seeing it
    // ai: still verified) would have done under the first's file.
    private fun keep(r: Readout.Rx, root: String, type: String) {
        if (filing.isNotEmpty() || root == unfiled || files.any { it.root == root }) return
        filing = root
        val dest = library.dest(root, r.name)
        engine.keep(dest) { ok ->
            filing = ""
            if (ok) { library.kept(dest, r.name, type, r.size, root); files = library.list() }
            else { unfiled = root; dest.parentFile?.deleteRecursively() }
        }
    }

    private val replaysDir get() = File(filesDir, "replays")

    // ai: a run's folder deleted on a thread of its own, unless it is being ended or packed (whoever finishes with it
    // ai: deletes it then, if `runs` no longer lists it)
    private fun dropRun(r: Replay?) {
        if (r == null || r.run in busyRuns) return
        Thread({ r.dir.deleteRecursively() }, "lizard-replays").start()
    }
    private fun putRun(r: Replay) { val i = runs.indexOfFirst { it.run == r.run }; if (i >= 0) runs[i] = r }

    // ai: The runs on disk at the process's first activity: the newest whole one (its meta.json written) is listed, with
    // ai: its Download; the rest go (one with no meta.json never ended: its process died first).
    private fun adoptReplays() {
        if (adopted) return
        adopted = true
        val runsOnDisk = replaysDir.listFiles()?.filter { it.isDirectory }.orEmpty().sortedBy { it.name }
        val whole = runsOnDisk.lastOrNull { File(it, "meta.json").isFile }
        if (whole != null) {
            val meta = runCatching { JSONObject(File(whole, "meta.json").readText()) }.getOrNull()
            val bytes = whole.listFiles()?.filter { it.name.endsWith(".gray") }?.sumOf { it.length() } ?: 0L
            runs.add(Replay(whole.name, whole, 0L, ReplayState.Ready, meta?.optLong("frames") ?: 0L, bytes, meta?.optString("error").orEmpty()))
        }
        val rest = runsOnDisk.filter { it != whole }
        if (rest.isNotEmpty()) Thread({ rest.forEach { it.deleteRecursively() } }, "lizard-replays").start()
    }

    // ai: A replay begun (the camera running with Save replays on). The finished run before it stays, with its Download,
    // ai: while this one records, and goes when a newer run ends with frames (the rule above; the deletion moved from a
    // ai: run's start to its end on 2026-10-03: Receive starts its camera by itself on a return,
    // ai: which began a run and took the last one before anyone could download it). A failed run leaves the list now (its
    // ai: folder went when it failed), and every folder no run names, but for one being ended or packed. The run named as
    // ai: the web names one, a millisecond on where the name is taken.
    private fun beginReplay() {
        runs.removeAll { it.state == ReplayState.Failed }
        val named = runs.map { it.run }.toSet()
        val old = replaysDir.listFiles()?.filter { it.name !in named && it.name !in busyRuns }.orEmpty()
        if (old.isNotEmpty()) Thread({ old.forEach { it.deleteRecursively() } }, "lizard-replays").start()
        val fmt = SimpleDateFormat("yyyy-MM-dd'T'HH-mm-ss-SSS'Z'", Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }
        var t = System.currentTimeMillis()
        var run: String
        while (true) { run = "run-" + fmt.format(Date(t)); if (run !in named && run !in busyRuns && !File(replaysDir, run).exists()) break; t++ }
        val dir = File(replaysDir, run)
        val h = Native.replayNew(dir.path, run, REPLAY_FRAMES)
        runs.add(Replay(run, dir, h, ReplayState.Recording, owner = engine))
        statsRows.clear()   // ai: a run's rows are its own
        engine.replay(h)
        Log.i(Engine.TAG, "replay: $run")
    }

    // ai: The run's end (the switch off, the camera stopped, the activity gone): the frames the decoder still holds for it
    // ai: come in first, then its frames are renamed into order and its meta written (the receiver's fields, then the
    // ai: camera and the settings as they are at its end), off the main thread. Ended with frames, it keeps its
    // ai: Download and every finished run begun before it goes; failed, or with no frame (the camera stopped before the
    // ai: decoder handed one over), it goes and the runs before it stay.
    private fun endReplay(r: Replay) {
        busyRuns.add(r.run)
        putRun(r.copy(state = ReplayState.Ending, owner = null))
        val c = cam
        val cn = engine.cameraNow()
        val more = JSONObject().apply {
            if (c != null) put("camera", JSONObject().put("id", c.id).put("format", c.format).put("size", "${c.size.width}x${c.size.height}")
                .put("fps", "[${c.fps.lower},${c.fps.upper}]").put("minFrameMs", c.minFrameMs).put("sensorOrientation", c.sensorOrientation).put("note", c.note)
                .apply { if (cn != null) put("exposureMs", cn.exposureMs).put("iso", cn.iso).put("readoutMs", cn.readoutMs) })
            put("zoom", settings.zoom); put("batch", settings.batch); put("thermal", heat); put("app", aboutLine())
        }.toString()
        engine.replayEnd(r.handle, raw, statsRows.joinToString("\n"), more) { res ->
            busyRuns.remove(r.run)
            val done = res.fold({ s ->
                val j = JSONObject(s)
                val n = j.optLong("frames")
                if (n > 0) r.copy(handle = 0L, state = ReplayState.Ready, frames = n, bytes = j.optLong("bytes"), why = j.optString("error"), owner = null)
                else r.copy(handle = 0L, state = ReplayState.Failed, why = j.optString("error").ifEmpty { "no frames" }, owner = null)
            }, { r.copy(handle = 0L, state = ReplayState.Failed, why = it.message.orEmpty(), owner = null) })
            Log.i(Engine.TAG, "replay: ${r.run} " + if (done.state == ReplayState.Ready) "${done.frames} frames, ${done.bytes} B${if (done.why.isEmpty()) "" else ", ${done.why}"}" else "failed: ${done.why}")
            if (runs.none { it.run == r.run }) { dropRun(done); return@replayEnd }
            putRun(done)
            if (done.state != ReplayState.Ready) { dropRun(done); return@replayEnd }
            val newest = runs.indexOfLast { it.state == ReplayState.Ready }
            val gone = runs.filterIndexed { i, it -> i < newest && (it.state == ReplayState.Ready || it.state == ReplayState.Failed) }
            runs.removeAll(gone)
            gone.forEach { dropRun(it) }
        }
    }

    // ai: Download: the replay packed into one file, Download/<run>.tar.gz (ReplayPack.kt: the web's tar, gzip at
    // ai: level 1), on a thread of its own: through the file system from Android 11, through MediaStore's Downloads
    // ai: on Android 10, where a path in Download/ cannot be written (no permission needed from 29). The run stays
    // ai: until it is superseded, and goes once packed if it was meanwhile.
    internal fun downloadReplay(r: Replay) {
        if (r.state != ReplayState.Ready || exporting.isNotEmpty() || !busyRuns.add(r.run)) return
        exporting = r.run
        val resolver = applicationContext.contentResolver
        Thread({
            val name = "${r.run}.tar.gz"
            val res = runCatching {
                if (!File(r.dir, "meta.json").isFile) throw IOException("the replay is gone")
                if (Build.VERSION.SDK_INT >= 30) {
                    @Suppress("DEPRECATION") val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                    packReplay(r.dir, File(downloads, name))
                } else {
                    val values = ContentValues().apply {
                        put(MediaStore.Downloads.DISPLAY_NAME, name); put(MediaStore.Downloads.MIME_TYPE, "application/gzip"); put(MediaStore.Downloads.IS_PENDING, 1)
                    }
                    val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: throw IOException("Downloads refused $name")
                    try {
                        val n = (resolver.openOutputStream(uri) ?: throw IOException("Downloads refused $name")).use { packReplay(r.dir, it) }
                        resolver.update(uri, ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }, null, null)
                        n
                    } catch (e: Throwable) {
                        runCatching { resolver.delete(uri, null, null) }
                        throw e
                    }
                }
            }
            main.post {
                busyRuns.remove(r.run)
                exporting = ""
                replayNote = res.fold({ "$name in Downloads, ${"%.0f".format(Locale.ROOT, it / 1e6)} MB\nthen: adb pull " +
                    "/sdcard/Download/$name && mkdir -p research/captures/v0.3/${r.run} && tar -xzf $name -C research/captures/v0.3/${r.run}" },
                    { "${r.run}: ${it.message}" })
                if (runs.none { it.run == r.run }) dropRun(r)
                Log.i(Engine.TAG, "replay: ${r.run} ${res.fold({ "downloaded" }, { "not downloaded: ${it.message}" })}")
            }
        }, "lizard-download").start()
    }

    @Composable
    private fun App() {
        LaunchedEffect(Unit) {
            var tick = 0
            while (true) {
                val s = withContext(Dispatchers.Default) { engine.stats() }
                raw = s
                val j = if (s.isEmpty()) null else runCatching { JSONObject(s) }.getOrNull()
                val r = rxOf(j)
                rx = r
                if (settings.decoder == "auto" && phase == Engine.Phase.On) j?.optString("decoder").orEmpty().let { if (it.isNotEmpty()) autoRan = if (it.startsWith("gpu")) "GPU" else "CPU" }
                val f = j?.optJSONObject("file")
                root = f?.optString("root").orEmpty()
                var last = false   // ai: this poll stopped the camera: its row posted now, the last a stopped camera sends
                if (r.verified && root.isNotEmpty()) {
                    keep(r, root, f?.optString("type").orEmpty())
                    // ai: The camera off once the file is in (2026-10-01): nothing is left to read, and the phone
                    // ai: cools. Once a root: the receiver holds the
                    // ai: verified file after the stop, so Start camera after it reads on rather than stopping again.
                    if (root != ended) {
                        ended = root
                        if (phase == Engine.Phase.On || phase == Engine.Phase.Starting || phase == Engine.Phase.Loading) { stopped = true; engine.stop(); last = true }
                    }
                }
                // ai: the phone's power once a second (readPower), its 5 s mean in the log; the dim switch
                if (tick % 4 == 0) {
                    readPower()
                    if (tick % 20 == 0 && powerN > 0) { Log.i(Engine.TAG, "power: %.2f W (5 s mean), battery %.1f C".format(powerSum / powerN, batteryC)); powerSum = 0.0; powerN = 0 }
                    else if (tick % 20 == 0) Log.i(Engine.TAG, "power: plugged in, battery %.1f C".format(batteryC))
                    val d = Native.prop("debug.lizard.dim")
                    val on = phase == Engine.Phase.On
                    val want = if (on) d.toFloatOrNull()?.takeIf { it in 0f..1f } else null
                    val key = if (want == null) "" else d
                    if (key != dimAsked) {
                        dimAsked = key
                        window.attributes = window.attributes.apply { screenBrightness = want ?: WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE }
                        Log.i(Engine.TAG, "dim: window brightness " + (want?.toString() ?: "the system's"))
                    }
                }
                // ai: the GPU's clock once a second, a log line at the first read and whenever a throttle's ceiling moves
                if (tick % 4 == 0) {
                    val c = withContext(Dispatchers.IO) { Clocks.read() }
                    val was = clocks
                    clocks = c
                    if (c != null && (was == null || ((c.throttled || was.throttled) && c != was)))
                        Log.i(Engine.TAG, "clocks: gpu ${c.mhz} of ${c.top} MHz, thermal $heat")
                }
                // ai: a stats row once a second (4 polls) while a replay records, for its stats.jsonl, and at no other time
                // ai: (2026-10-09: rows are logged only as part of a replay); the transfer's average since its first decode added to it
                val recording = runs.any { it.state == ReplayState.Recording && it.owner === engine }
                if ((tick++ % 4 == 0 || last) && recording && s.isNotEmpty() && (phase == Engine.Phase.On || last)) {
                    // ai: and the phone's thermal status (`thermal`, 2026-10-01), so a slower camera can be told from a hot one;
                    // ai: the GPU's clock ceiling (`gpuMaxMHz`, 2026-10-03); the camera's exposure, sensitivity, readout and
                    // ai: frame, and the phase lock's snapshot (`phase`: its arm and
                    // ai: standing, the leaks it holds at, its gain and pace; 2026-10-04, so a replay says what the
                    // ai: capture's window was and what the lock made of it)
                    val c = clocks
                    val cn = engine.cameraNow(); val ph = engine.phaseState()
                    val body = runCatching { JSONObject(s).put("thermal", heat).apply {
                        if (c != null) put("gpuMaxMHz", c.mhz)
                        powerW?.let { put("powerW", it) }; put("batteryC", batteryC)
                        if (cn != null) { put("exposureMs", cn.exposureMs); put("iso", cn.iso); put("readoutMs", cn.readoutMs); put("frameMs", cn.frameMs) }
                        if (ph != null) put("phase", JSONObject().put("arm", ph.arm).put("state", ph.what).put("a", nz(ph.a)).put("b", nz(ph.b)).put("k", nz(ph.k))
                            .put("se", nz(ph.se)).put("n", ph.n).put("gain", ph.gain).put("pace", ph.pace).put("stood", ph.stood).put("edge", ph.edge).put("flatMs", ph.flatMs).put("delays", ph.delays))
                        if (j != null && r.hasFile && !r.verified && r.secs >= 1) put("avgKBs", r.received / r.secs / 1000) }.toString() }.getOrDefault(s)
                    statsRows.addLast(body)
                    while (statsRows.size > REPLAY_LOG) statsRows.removeFirst()
                }
                // ai: Save replays: a run while the switch is on and this activity's camera runs, ended when either stops
                val mine = runs.lastOrNull { it.state == ReplayState.Recording && it.owner === engine }
                val replaying = settings.replays == "on" && phase == Engine.Phase.On
                if (replaying && mine == null) beginReplay()
                else if (!replaying && mine != null) endReplay(mine)
                delay(250)
            }
        }
        BackHandler(enabled = screen != Screen.Home) {
            when {
                screen == Screen.Send && send.phase != SendState.Phase.Idle && send.phase !is SendState.Phase.Error -> send.stop()
                else -> go(Screen.Home)
            }
        }
        when (screen) {
            Screen.Home -> HomeScreen()
            Screen.Receive -> ReceiveScreen()
            Screen.Send -> SendScreen()
        }
    }
}

// ai: The stats JSON's fields the screens read (receiver_stub.cpp's header lists them).
internal fun rxOf(j: JSONObject?): Readout.Rx {
    if (j == null) return Readout.Rx()
    val f = j.optJSONObject("file")
    return Readout.Rx(
        state = j.optString("state"), error = j.optString("error"), goodputKBs = j.optDouble("goodputKBs", 0.0),
        foundShare = j.optDouble("foundShare", 0.0), bandVersion = j.optInt("bandVersion", 0), fps = j.optJSONObject("word")?.optInt("fps", 0) ?: 0,
        hasFile = f != null, name = f?.optString("name").orEmpty().ifEmpty { "a file" }, size = f?.optLong("size", 0) ?: 0,
        received = f?.optLong("received", 0) ?: 0, verified = f?.optBoolean("verified", false) ?: false,
        sent = f?.optLong("sent", 0) ?: 0, sentIn = f?.optLong("sentIn", 0) ?: 0, secs = f?.optDouble("secs", 0.0) ?: 0.0,
    )
}
