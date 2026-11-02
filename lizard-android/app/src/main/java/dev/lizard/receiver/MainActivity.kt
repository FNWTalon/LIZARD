package dev.lizard.receiver

import dev.lizard.readout.Readout
import android.Manifest
import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.Context
import android.content.BroadcastReceiver
import android.content.ContentValues
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.pm.ShortcutInfo
import android.content.pm.ShortcutManager
import android.graphics.drawable.Icon
import android.net.Uri
import android.provider.Settings as AndroidSettings
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.os.PowerManager
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.util.Log
import android.view.WindowManager
import android.hardware.usb.UsbAccessory
import android.hardware.usb.UsbManager
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import dev.lizard.ui.*
import androidx.compose.ui.unit.dp

// ai: The app (2026-10-01): three screens, Home (Send and Receive, the received files, and at the bottom Settings: the
// ai: tips again, the received files' size, About), Receive (the camera and the transfer, then Settings and Developer
// ai: Tools, rows that open, and About) and Send (a file, Start, Settings, Developer Tools, About); since 2026-10-02 as
// ai: the web's pages after their tidying. One activity, its `screen` and the system back between them, no navigation
// ai: library. The web's pages are the same shell (lizard-web/index.html, recv.html, send.html) and the words the same
// ai: (Readout.kt). The camera runs on Receive alone, from entering it (or a resume there) to leaving it (or a pause, or
// ai: Stop camera); the receive session (ReceiveState, the engine's), and a transfer partly in, outlives it as it
// ai: outlives Stop. The launcher opens Home; `--es screen receive` (tools/phone/live.sh, arms.sh) and the launcher
// ai: shortcut (publishShortcut) open Receive with the camera, in this activity when it runs (singleTop, onNewIntent).
// ai: Received files land in Downloads (Received.kt, since 2026-10-31; the app's own library before).
// ai: the phone's target size: buttons, fields and icon buttons 48 dp, the web's under a finger
val TARGET = 48.dp

// ai: About's credits for what the app carries besides the core (the repository's NOTICE)
private val CREDITS_HERE = listOf(
    Credit("AndroidX", "Apache License 2.0"),
    Credit("Jetpack Compose", "Apache License 2.0"),
    Credit("Kotlin", "Apache License 2.0"),
)

class MainActivity : ComponentActivity() {
    companion object {
        const val UNPLUG_MS = 2000L   // ai: a cable out this long is out (usb)
        const val PC_GONE_MS = 2000L  // ai: a PC decoder with no PC heard this long, past the link's QUIET_MS, lets go
        // ai: the folds' keys, and the two collapsers' (Receive's column, the sending screen's panel; landscape)
        val FOLDS = listOf("recvSettings", "recvAdvanced", "recvExperimental", "sendSettings", "sendAdvanced", "sendExperimental", "recvCollapsed", "sendCollapsed")
        // ai: the received files' own work, one at a time in the order asked (placing into Downloads, deletes)
        private val filer = java.util.concurrent.Executors.newSingleThreadExecutor { r -> Thread(r, "lizard-files") }
        // ai: Save replays' Download: the run a copy is under way for, and how the last went
        internal var exporting by mutableStateOf("")
        internal var replayNote by mutableStateOf("")
    }

    internal lateinit var receive: ReceiveState
    internal val home by lazy { AndroidHome(this) }
    internal val receiveUi by lazy { AndroidReceive(this) }
    internal val sendUi by lazy { AndroidSend(this) }
    internal lateinit var received: Received
    private lateinit var prefs: SharedPreferences
    internal val phase get() = receive.phase
    internal val cam get() = receive.cam
    internal val stats get() = receive.stats
    internal var settings by mutableStateOf(Settings())
    internal var screen by mutableStateOf(Screen.Home)
    internal var stopped by mutableStateOf(false)          // ai: Receive: Stop camera pressed, the camera off until Start
    internal var granted by mutableStateOf(false)
    internal var denied by mutableStateOf(false)           // ai: refused, and Android asks no more: only its settings allow it
    internal var note by mutableStateOf("")                // ai: what came of a Save a copy or an Open, or a file Downloads did not take
    internal var files by mutableStateOf(listOf<Received.Entry>())
    // ai: transfers begun and not finished (2026-10-22, the session's store): Home lists them to continue or delete
    internal var partials by mutableStateOf(listOf<Partial>())
    internal var tipsSeen by mutableStateOf(false)
    // ai: The landscape column's width on Receive and Send, dp (2026-10-05): 0 the default (SideWidth), else what its edge
    // ai: was dragged to (SideEdge), kept as `sideDp`; sideDragging while a finger is on the edge (Send re-picks after).
    internal var sideDp by mutableStateOf(0f)
    internal var sideDragging by mutableStateOf(false)
    internal fun saveSide() { prefs.edit().putFloat("sideDp", sideDp).apply() }
    // ai: the rows that open, open or not, kept by key (the web's send:dev, send:logs, recv:dev, recv:logs)
    internal val folds = mutableStateMapOf<String, Boolean>()
    internal var autoRan by mutableStateOf("")             // ai: what auto last decoded on, GPU or CPU (the Decoder menu's value under auto)
    internal var hint by mutableStateOf("")                // ai: the one hint while the camera reads (ReceiveState.hint)
    // ai: The phone's USB and the PC link (2026-10-27): while a cable is in (USB_STATE's connected, while this activity
    // ai: runs) the core's PC link is up (the session's link: the listener, the beacon, the accessory's sessions),
    // ai: answering a PC whatever the decoder; a cable out for UNPLUG_MS lets it go (the drop as the PC switches the
    // ai: phone into accessory mode does not). The PC decoder is offered while a PC receiver is heard on it (pcPresent),
    // ai: and a PC unheard for PC_GONE_MS past the link's own QUIET_MS takes a PC decoder back to auto and the camera off.
    internal var usb by mutableStateOf(PcLink.Usb(false, false))
    internal val link get() = receive.link
    internal val pcPresent get() = link?.flag("present") == true
    private var pcWant = 0L       // ai: the PC's Start and Stop as last followed (the link's want.seq)
    private var pcMissedAt = 0L   // ai: when a PC decoder first went without a PC heard (elapsedRealtime; 0 none)
    private val unplugged = Runnable {
        if (!PcLink.usb(this).connected) { Log.i(TAG, "usb: the cable out, the PC link down"); receive.linkDown() }
    }
    private val usbListener = object : BroadcastReceiver() {
        override fun onReceive(c: Context, i: Intent) {
            usb = PcLink.usbOf(i)
            main.removeCallbacks(unplugged)
            if (usb.connected) receive.linkUp(settings.layout) else main.postDelayed(unplugged, UNPLUG_MS)
        }
    }
    internal var rx by mutableStateOf(Readout.Rx())
    internal var root by mutableStateOf("")                // ai: the transfer in hand's BLAKE3 root, hex
    private var ended = ""                                 // ai: the root whose arrival stopped the camera (once a root)
    private var saving: Received.Entry? = null
    private var savingTree: Received.Entry? = null
    internal var shownBundle by mutableStateOf<Received.Entry?>(null)   // ai: the bundle the Files screen lists
    private var asked = false
    private val main = Handler(Looper.getMainLooper())

    private val ask = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        granted = ok
        denied = !ok && !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)
        if (ok && wantsCamera() && resumed()) startCamera()
    }
    // ai: the app's sender (Send.kt) and the choosers it opens: files, one or several, or a folder (2026-10-16)
    internal val send by lazy { SendState(this) }
    private val choose = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { us -> if (us.isNotEmpty()) send.pick(us) }
    internal fun pickFile() { try { choose.launch(arrayOf("*/*")) } catch (_: ActivityNotFoundException) {} }
    private val chooseTree = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { u -> if (u != null) send.pickTree(u) }
    internal fun pickFolder() { try { chooseTree.launch(null) } catch (_: ActivityNotFoundException) {} }
    // ai: photos and videos through the system's photo picker (2026-10-23): no permission, each one a file item
    private val chooseMedia = registerForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { us -> if (us.isNotEmpty()) send.pick(us) }
    internal fun pickMedia() { try { chooseMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo)) } catch (_: ActivityNotFoundException) {} }
    private val save = registerForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        val e = saving
        saving = null
        if (uri != null && e != null) saveTo(uri, e)
    }
    // ai: a bundle's Save: the folder where the user picks, made again there file by file (saveTreeTo)
    private val saveTree = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        val e = savingTree
        savingTree = null
        if (uri != null && e != null) saveTreeTo(uri, e)
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
        settings = Settings.load(this)
        val data = File(filesDir, "session")
        carryOver(data)
        receive = ReceiveState(this)
        received = Received(this, data)
        receive.linkPack(settings.pcPack != "off")
        refile()
        refreshPartials()
        granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        // ai: a restored activity (a configuration change, the process gone and back) keeps the screen it was on; only a
        // ai: fresh one takes its launch intent's (2026-10-01: restored, it went back to the intent's Receive)
        savedInstanceState?.getString("screen")?.let { s -> Screen.entries.firstOrNull { it.name == s }?.let { screen = it } }
        if (savedInstanceState == null) handle(intent)
        publishShortcut()
        // ai: a mouse's pointer over a column's edge (DeX, a mouse on a tablet): Android's left-right double arrow
        setContent { LizardTheme(TARGET, resizeIcon = PointerIcon(android.view.PointerIcon.TYPE_HORIZONTAL_DOUBLE_ARROW), credits = CREDITS_HERE) { App() } }
    }

    // ai: What a phone held before the receive session (2026-10-31), carried into its data folder at the first start, one
    // ai: rename each (one filesystem), nothing lost: the store (filesDir/received/lizard-xfer: unfinished transfers
    // ai: resume) to <data>/store, the received files (filesDir/library) to <data>/library, which Received places into
    // ai: Downloads, Save replays' runs (filesDir/replays) to <data>/replays, which the session adopts, and the cameras'
    // ai: settings (the preferences' resolution@, format@, zoom@) into <data>/cameras.json where it is missing.
    private fun carryOver(data: File) {
        data.mkdirs()
        fun move(from: File, to: File) {
            if (!from.exists()) return
            to.parentFile?.mkdirs()
            if (!to.exists() && from.renameTo(to)) { Log.i(TAG, "carried: ${from.path} to ${to.path}"); return }
            // ai: both there: each entry the new one lacks moved in, the rest left where it was
            for (f in from.listFiles().orEmpty()) { val t = File(to, f.name); if (!t.exists() && !f.renameTo(t)) Log.w(TAG, "carried: ${f.path} not moved") }
            if (from.listFiles().isNullOrEmpty()) from.delete()
        }
        move(File(filesDir, "received/lizard-xfer"), File(data, "store/lizard-xfer"))
        File(filesDir, "received").let { if (it.isDirectory && it.listFiles().isNullOrEmpty()) it.delete() }
        move(File(filesDir, "library"), File(data, "library"))
        move(File(filesDir, "replays"), File(data, "replays"))
        val cams = File(data, "cameras.json")
        if (!cams.exists()) Settings.camerasMoved(this)?.let { j ->
            runCatching { val tmp = File(data, "cameras.json.tmp"); tmp.writeText(j); if (!tmp.renameTo(cams)) throw IOException("not renamed") }
                .onSuccess { Log.i(TAG, "carried: the cameras' settings into ${cams.path}") }.onFailure { Log.w(TAG, "carried: ${cams.path}: ${it.message}") }
        }
    }

    override fun onNewIntent(intent: Intent) { super.onNewIntent(intent); handle(intent) }
    // ai: Receive from the shortcut or `--es screen receive`; Send from its shortcut, or with files another app shared to LIZARD (one,
    // ai: or several since 2026-10-16: one transfer, as the sender bundles them)
    private fun handle(i: Intent?) {
        // ai: the PC's USB accessory (PcLink): opened, and Receive, where the PC decoder serves it
        if (i?.action == UsbManager.ACTION_USB_ACCESSORY_ATTACHED) {
            @Suppress("DEPRECATION") val acc = i.getParcelableExtra<UsbAccessory>(UsbManager.EXTRA_ACCESSORY)
            PcLink.openAccessory(this, acc)
            go(Screen.Receive)
            return
        }
        if (i?.action == Intent.ACTION_SEND || i?.action == Intent.ACTION_SEND_MULTIPLE) {
            @Suppress("DEPRECATION") val us = if (i.action == Intent.ACTION_SEND) listOfNotNull(i.getParcelableExtra<Uri>(Intent.EXTRA_STREAM))
                else i.getParcelableArrayListExtra<Uri>(Intent.EXTRA_STREAM).orEmpty()
            if (us.isNotEmpty()) send.pick(us)
            go(Screen.Send)
            return
        }
        when (i?.getStringExtra("screen")) { "receive" -> go(Screen.Receive); "send" -> go(Screen.Send) }
    }
    // ai: The launcher icon's long-press shortcut, Receive: dynamic, so its intent carries the flags given here. A static
    // ai: one (res/xml/shortcuts.xml until 2026-10-01) is always started with FLAG_ACTIVITY_CLEAR_TASK, which finished
    // ai: this activity, its session and a transfer partly in, and made another beside it. From here the
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
    }.onFailure { Log.w(TAG, "shortcut: ${it.message}") }
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
    // ai: plugged in, when it measures the charger instead; read once a second, the stats' `powerW` and `batteryC`
    // ai: (the session's extra, so every replay row has them), a `power:` log line every 5 s with the mean (and the
    // ai: thermal headroom, headroom()).
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
    // ai: the thermal headroom 10 s ahead for the power line (2026-10-26: whether a live run's slow seconds were the phone's
    // ai: heat, which the status says only in steps): PowerManager's, Android 11 and up, 1.0 where the phone throttles;
    // ai: "" where it gives none (NaN when asked again within a second, or unsupported)
    private fun headroom(): String {
        if (Build.VERSION.SDK_INT < 30) return ""
        val h = getSystemService(PowerManager::class.java)?.getThermalHeadroom(10) ?: return ""
        return if (h.isNaN()) "" else ", thermal headroom %.2f".format(h)
    }
    // ai: the phone's own fields for the session's stats and replay rows (session.h extra): the thermal status, the GPU's
    // ai: clock ceiling, the power off the charger, the battery's temperature
    private fun platform(): String = buildJsonObject {
        put("thermal", heat)
        clocks?.let { put("gpuMaxMHz", it.mhz) }
        powerW?.let { put("powerW", it) }
        put("batteryC", batteryC)
    }.toString()
    private val heatListener = PowerManager.OnThermalStatusChangedListener { if (it != heat) Log.i(TAG, "thermal status $it"); heat = it }
    override fun onResume() {
        super.onResume()
        Log.i(TAG, "resume")
        getSystemService(PowerManager::class.java)?.addThermalStatusListener(mainExecutor, heatListener)
        // ai: the permission again: it may have been given in Android's settings meanwhile
        granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (granted) denied = false
        val usbFilter = IntentFilter(PcLink.USB_STATE)
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(usbListener, usbFilter, Context.RECEIVER_EXPORTED) else registerReceiver(usbListener, usbFilter)
        PcLink.openAccessory(this)
        if (wantsCamera()) startCamera()
        else if (screen == Screen.Receive && !granted && !asked) askCamera()
    }
    override fun onPause() {
        super.onPause(); Log.i(TAG, "pause"); receive.stop(); send.stop()
        runCatching { unregisterReceiver(usbListener) }
        main.removeCallbacks(unplugged)
        getSystemService(PowerManager::class.java)?.removeThermalStatusListener(heatListener)
    }
    override fun onDestroy() { super.onDestroy(); receive.close() }

    private fun resumed() = lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)
    private fun wantsCamera() = screen == Screen.Receive && !stopped && granted
    internal fun startCamera() = receive.start()
    internal val running get() = phase == RxPhase.On || phase == RxPhase.Starting || phase == RxPhase.Loading

    // ai: the screen while the camera runs: kept on, and full screen with the system bars hidden from the camera's start
    // ai: until it stops (2026-10-05, as Send does while sending: SendState.screenFor); leaving Receive stops the camera,
    // ai: which brings them back
    internal fun onPhase(p: RxPhase) {
        if (p == RxPhase.On) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else if (p !is RxPhase.Starting && p !is RxPhase.Loading) window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val bars = WindowCompat.getInsetsController(window, window.decorView)
        if (p == RxPhase.On || p is RxPhase.Starting || p is RxPhase.Loading) bars.hide(WindowInsetsCompat.Type.systemBars())
        else bars.show(WindowInsetsCompat.Type.systemBars())
    }

    // ai: Between screens. Leaving Receive stops the camera; entering it starts it (only while resumed: onResume starts it
    // ai: otherwise, so a launch into Receive opens the camera once), asking for the permission the first time.
    internal fun go(s: Screen) {
        if (s == screen) return
        if (screen == Screen.Receive) receive.stop()
        if (screen == Screen.Send) send.stop()
        screen = s
        note = ""
        if (s == Screen.Home) { refreshFiles(); refreshPartials() }
        if (s == Screen.Receive) {
            stopped = false
            if (resumed()) { if (granted) startCamera() else askCamera() }
        }
    }
    internal fun toggleCamera() {
        if (running) { stopped = true; receive.stop() }
        else { stopped = false; startCamera() }
    }
    internal fun askCamera() { asked = true; ask.launch(Manifest.permission.CAMERA) }
    internal fun openAppSettings() {
        try { startActivity(Intent(AndroidSettings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null))) } catch (_: ActivityNotFoundException) {}
    }

    // ai: A Settings or Developer Tools switch, kept, and handed to the session: another decoder or crop a new
    // ai: receiver on the transfer in hand (the camera started again around it); another camera opened (started where
    // ai: the camera reads); the batch set live; the PC link's crop and packing.
    internal fun change(s: Settings) {
        val old = settings
        settings = s
        s.save(this)
        if (s.layout != old.layout && usb.connected) receive.linkUp(s.layout)   // ai: the crop the PC's hello names
        if (s.pcPack != old.pcPack) receive.linkPack(s.pcPack != "off")         // ai: from the next frame queued on
        if (s.frames != old.frames) receive.batch(s.frames)
        if (s.decoder != old.decoder && s.decoder != "auto") autoRan = ""
        if (s.decoder != old.decoder || s.layout != old.layout) receive.reconfigure()
        if (s.camera != old.camera) receive.cameraChosen(running)
        replaysAsked()
    }
    // ai: Save replays: the session's switch, on with the switch and any decoder but the PC's (which keeps no frames: the
    // ai: PC's own Save replays records them)
    private var replaysOn = false
    private fun replaysAsked() {
        val on = settings.replays == "on" && settings.decoder != "pc"
        if (on != replaysOn) { replaysOn = on; receive.replays(on) }
    }
    internal fun tipsDone() { tipsSeen = true; prefs.edit().putBoolean("tipsSeen", true).apply() }
    internal fun tipsAgain() { tipsSeen = false; prefs.edit().putBoolean("tipsSeen", false).apply() }
    internal fun isOpen(k: String) = folds[k] == true
    internal fun toggle(k: String) { val v = !isOpen(k); folds[k] = v; prefs.edit().putBoolean(k, v).apply() }

    // ai: The received files' actions: a file through its own URI (MediaStore's in Downloads, the FileProvider's for one
    // ai: Downloads did not take), as the header's media type. A bundle's (2026-10-16): Open lists its files (the Files
    // ai: screen), Share hands them all over at once, Save makes its folder again where the user picks; a file in it
    // ai: opens and shares as one alone, by its extension's type.
    internal fun open(e: Received.Entry) { if (e.bundle) openBundle(e) else openUri(received.uriOf(e.uri.ifEmpty { e.local }), e.mime) }
    internal fun openUri(u: Uri, mime: String) {
        val i = Intent(Intent.ACTION_VIEW).setDataAndType(u, mime).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        try { startActivity(i) } catch (_: ActivityNotFoundException) { note = "No app on this phone opens $mime files: share it, or save a copy." }
    }
    internal fun openBundle(e: Received.Entry) { shownBundle = e; go(Screen.Files) }
    internal fun share(e: Received.Entry) { if (e.bundle) shareAll(e) else shareUri(received.uriOf(e.uri.ifEmpty { e.local }), e.name, e.mime) }
    internal fun shareUri(uri: Uri, name: String, mime: String) {
        val i = Intent(Intent.ACTION_SEND).setType(mime).putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        i.clipData = ClipData.newRawUri(name, uri)
        try { startActivity(Intent.createChooser(i, null)) } catch (_: ActivityNotFoundException) {}
    }
    // ai: every file of a bundle in one share, the type they have in common ("image/*", else any); a share too large
    // ai: for Android's binder (thousands of files) is said, and Save does it
    private fun shareAll(e: Received.Entry) {
        if (e.items.isEmpty()) { note = "${e.name} holds no files."; return }
        val uris = ArrayList(e.items.map { received.uriOf(it.uri) })
        val types = e.items.map { Received.mimeOf(it.path) }.distinct()
        val kinds = types.map { it.substringBefore('/') }.distinct()
        val type = types.singleOrNull() ?: kinds.singleOrNull()?.let { "$it/*" } ?: "*/*"
        val i = Intent(Intent.ACTION_SEND_MULTIPLE).setType(type).putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        i.clipData = ClipData.newRawUri(e.name, uris[0]).apply { for (u in uris.drop(1)) addItem(ClipData.Item(u)) }
        try { startActivity(Intent.createChooser(i, null)) } catch (_: ActivityNotFoundException) {} catch (_: RuntimeException) { note = "Too many files to share at once: save the folder instead." }
    }
    internal fun saveCopy(e: Received.Entry) {   // ai: the Save buttons (a copy where the user picks; a bundle's folder into a folder picked)
        note = ""
        if (e.bundle) { savingTree = e; try { saveTree.launch(null) } catch (_: ActivityNotFoundException) {} }
        else { saving = e; save.launch(e.name) }
    }
    private fun saveTreeTo(tree: Uri, e: Received.Entry) {
        Thread {
            val n = try {
                val cr = contentResolver
                val dirs = HashMap<String, Uri>()
                val top = DocumentsContract.createDocument(cr, DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree)),
                    DocumentsContract.Document.MIME_TYPE_DIR, e.name) ?: throw IOException("${e.name} could not be made")
                // ai: a file's folder made where it is not yet, its parents first
                fun dirOf(path: String): Uri {
                    if (path.isEmpty()) return top
                    dirs[path]?.let { return it }
                    val d = DocumentsContract.createDocument(cr, dirOf(path.substringBeforeLast('/', "")), DocumentsContract.Document.MIME_TYPE_DIR,
                        path.substringAfterLast('/')) ?: throw IOException("$path could not be made")
                    dirs[path] = d
                    return d
                }
                for (it in e.items.sortedBy { it.path }) {
                    val d = DocumentsContract.createDocument(cr, dirOf(it.path.substringBeforeLast('/', "")), Received.mimeOf(it.path), it.path.substringAfterLast('/'))
                        ?: throw IOException("${it.path} could not be made")
                    (cr.openOutputStream(d) ?: throw IOException("${it.path} could not be written")).use { o ->
                        (cr.openInputStream(received.uriOf(it.uri)) ?: throw IOException("${it.path} could not be read")).use { i -> i.copyTo(o, 1 shl 16) }
                    }
                }
                "Saved ${e.name}"
            } catch (x: Exception) { "The folder could not be saved: ${x.message}" }
            runOnUiThread { note = n }
        }.start()
    }
    private fun saveTo(uri: Uri, e: Received.Entry) {
        Thread {
            val n = try {
                contentResolver.openOutputStream(uri)!!.use { o ->
                    (contentResolver.openInputStream(received.uriOf(e.uri.ifEmpty { e.local })) ?: throw IOException("the file could not be read")).use { it.copyTo(o, 1 shl 16) }
                }
                "Saved ${e.name}"
            } catch (x: Exception) { "The copy could not be saved: ${x.message}" }
            runOnUiThread { note = n }
        }.start()
    }
    // ai: deleting the file the receiver shows as received forgets it there too (2026-10-07; the session's clear): the
    // ai: state line's green "Received" clears at the next poll, and the file in the light is received anew; the file
    // ai: goes from Downloads (this app's own rows)
    internal fun delete(e: Received.Entry) = deleteAll(listOf(e))
    internal fun deleteAll(es: List<Received.Entry> = files) {
        val gone = es.map { it.id }.toSet()
        files = files.filter { it.id !in gone }
        if (root.isNotEmpty() && es.any { it.root == root }) { receive.clear(); ended = "" }
        filer.execute {
            val left = es.mapNotNull { e -> runCatching { received.delete(e) }.exceptionOrNull()?.message }
            val l = received.list()
            main.post { files = l; left.firstOrNull()?.let { note = "Not deleted: $it" } }
        }
    }
    // ai: an unfinished transfer deleted (2026-10-22): the session forgets it (the receiver's, or its folder), or with no
    // ai: session its folder is deleted (ReceiveState.deletePartial); the list read again after
    internal fun deletePartial(p: Partial) {
        partials = partials.filter { it.id != p.id }
        receive.deletePartial(p) { refreshPartials() }
    }
    internal fun installed(): String = try {
        SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.ROOT).format(Date(packageManager.getPackageInfo(packageName, 0).lastUpdateTime))
    } catch (_: Exception) { "" }
    internal fun version(): String = try { packageManager.getPackageInfo(packageName, 0).versionName ?: "" } catch (_: Exception) { "" }

    // ai: Home's lists: the received files (MediaStore queried: off the main thread) and the unfinished transfers
    private fun refreshFiles() { filer.execute { val l = received.list(); main.post { files = l } } }
    private fun refreshPartials() { Thread({ val p = ReceiveState.partialsOf(receive.data); main.post { partials = p } }, "lizard-partials").start() }
    // ai: The takes in kept/ placed into Downloads (Received.retry), one pass queued at a time: at the start, and each
    // ai: time the session takes a file (its stats' kept.taken); a file Downloads did not take is said in the note
    private var filing = false
    private var refileAgain = false
    private fun refile() {
        if (filing) { refileAgain = true; return }
        filing = true
        filer.execute {
            val done = received.retry()
            val l = received.list()
            main.post {
                filing = false
                files = l
                done.firstOrNull { it.why.isNotEmpty() && (it.root == root || root.isEmpty()) }?.let { note = "Not saved to Downloads: ${it.why}" }
                if (refileAgain) { refileAgain = false; refile() }
            }
        }
    }

    // ai: Download: a run's .lizard copied to Download/<run>.lizard (what a PC's Upload camera plays, 2026-10-28), on a
    // ai: thread of its own: through the file system from Android 11 (a .part renamed at the end), through MediaStore's
    // ai: Downloads on Android 10, where a path in Download/ cannot be written (no permission needed from 29). A run the
    // ai: session lets go meanwhile (a newer one ended whole) fails the copy, said in the note.
    internal fun downloadReplay(r: Run) {
        if (r.state != RunState.Ready || exporting.isNotEmpty()) return
        exporting = r.run
        val resolver = applicationContext.contentResolver
        Thread({
            val name = "${r.run}.lizard"
            val res = runCatching {
                if (!r.file.isFile) throw IOException("the replay is gone")
                if (Build.VERSION.SDK_INT >= 30) {
                    @Suppress("DEPRECATION") val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                    val dest = File(downloads, name)
                    val part = File(downloads, "$name.part")
                    try {
                        r.file.copyTo(part, overwrite = true)
                        if (dest.exists() && !dest.delete()) throw IOException("$name could not be replaced")
                        if (!part.renameTo(dest)) throw IOException("${part.name} could not be renamed")
                    } catch (e: Throwable) { part.delete(); throw e }
                    dest.length()
                } else {
                    val values = ContentValues().apply {
                        put(MediaStore.Downloads.DISPLAY_NAME, name); put(MediaStore.Downloads.MIME_TYPE, "application/octet-stream"); put(MediaStore.Downloads.IS_PENDING, 1)
                    }
                    val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: throw IOException("Downloads refused $name")
                    try {
                        val n = (resolver.openOutputStream(uri) ?: throw IOException("Downloads refused $name")).use { out -> r.file.inputStream().use { it.copyTo(out) } }
                        resolver.update(uri, ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }, null, null)
                        n
                    } catch (e: Throwable) {
                        runCatching { resolver.delete(uri, null, null) }
                        throw e
                    }
                }
            }
            main.post {
                exporting = ""
                replayNote = res.fold({ "$name in Downloads, ${"%.0f".format(Locale.ROOT, it / 1e6)} MB\nthen: adb pull " +
                    "/sdcard/Download/$name research/replays/" }, { "${r.run}: ${it.message}" })
                Log.i(TAG, "replay: ${r.run} ${res.fold({ "downloaded" }, { "not downloaded: ${it.message}" })}")
            }
        }, "lizard-download").start()
    }

    @Composable
    private fun App() {
        LaunchedEffect(Unit) {
            var tick = 0
            var taken = -1L
            var made = -1        // ai: the session the takes were counted for (a new one counts from 0)
            while (true) {
                // ai: the phone's power and heat once a second, into the session's stats (and so every replay row)
                if (tick % 4 == 0) {
                    readPower()
                    if (tick % 20 == 0 && powerN > 0) { Log.i(TAG, "power: %.2f W (5 s mean), battery %.1f C%s".format(powerSum / powerN, batteryC, headroom())); powerSum = 0.0; powerN = 0 }
                    else if (tick % 20 == 0) Log.i(TAG, "power: plugged in, battery %.1f C%s".format(batteryC, headroom()))
                    val c = withContext(Dispatchers.IO) { Clocks.read() }
                    val was = clocks
                    clocks = c
                    if (c != null && (was == null || ((c.throttled || was.throttled) && c != was)))
                        Log.i(TAG, "clocks: gpu ${c.mhz} of ${c.top} MHz, thermal $heat")
                }
                val was = running
                val j = receive.poll(if (tick % 4 == 0) platform() else null)
                rx = rxOf(j)
                root = j?.obj("file")?.str("root").orEmpty()
                // ai: the camera off once the file is in (the session's rule, 2026-10-01: nothing is left to read, and the
                // ai: phone cools): held off here until Start, once a root (a Start after it reads on)
                if (was && phase == RxPhase.Idle && rx.verified && root.isNotEmpty() && root != ended) { ended = root; stopped = true }
                val lk = link
                // ai: Start or Stop pressed on the PC (liblizard/core/rx/stream.h CAMERA): followed once each, on Receive in
                // ai: front: Start takes the PC decoder where another runs, and the camera on; Stop the camera off. The
                // ai: phone's own Start and Stop reach the PC the same way.
                lk?.obj("want")?.let { w ->
                    val seq = w.num("seq").toLong()
                    if (seq != pcWant) {
                        pcWant = seq
                        val on = w.flag("on")
                        if (seq > 0 && screen == Screen.Receive && resumed() && (!on || granted)) {
                            if (on && settings.decoder != "pc") {
                                Log.i(TAG, "stream: the PC asks the camera on: the PC decoder")
                                stopped = false
                                change(settings.copy(decoder = "pc"))
                                if (!running) startCamera()
                            } else if (on != running) {
                                Log.i(TAG, "stream: the PC asks the camera ${if (on) "on" else "off"}")
                                toggleCamera()
                            }
                        }
                    }
                }
                // ai: no PC receiver heard (the PC's app closed, the PC off, the cable out): PC mode let go, the camera off
                if (settings.decoder == "pc" && !pcPresent) {
                    val now = SystemClock.elapsedRealtime()
                    if (pcMissedAt == 0L) pcMissedAt = now
                    else if (now - pcMissedAt >= PC_GONE_MS) {
                        pcMissedAt = 0L
                        Log.i(TAG, "stream: no PC receiver heard: the decoder back to auto, the camera off")
                        if (running) { stopped = true; receive.stop() }
                        change(settings.copy(decoder = "auto"))
                    }
                } else pcMissedAt = 0L
                if (settings.decoder == "auto" && phase == RxPhase.On) j?.str("autoRan").orEmpty().let { if (it.isNotEmpty()) autoRan = it }
                // ai: a file the session took into kept/ (it stopped the camera itself): placed into Downloads; a take that
                // ai: failed said in the note
                val k = j?.obj("kept")
                if (k != null && receive.made != made) { made = receive.made; taken = -1L }
                val n = k?.num("taken")?.toLong() ?: taken
                if (k != null && n != taken) { val first = taken < 0; taken = n; if (!first || n > 0) refile() }
                // ai: a take that failed said, and unsaid once a later try took it (the session's error cleared)
                k?.str("error")?.let { e -> if (e.isNotEmpty() && k.str("root") == root) note = "Not saved: $e" else if (e.isEmpty() && note.startsWith("Not saved: ")) note = "" }
                // ai: the unfinished transfers on Home, once a second (a transfer in hand writes its meta.json as it goes)
                if (tick % 4 == 0 && screen == Screen.Home) refreshPartials()
                replaysAsked()
                hint = receive.hint(j)
                tick++
                delay(250)
            }
        }
        BackHandler(enabled = screen != Screen.Home) {
            when {
                screen == Screen.Send && send.phase != SendState.Phase.Idle && send.phase !is SendState.Phase.Error -> send.stop()
                else -> go(Screen.Home)
            }
        }
        Screens(screen, home, sendUi, receiveUi) {
            shownBundle?.let { e -> BundleScreen(e.name, e.files, e.size, home.bundleActs, home.bundleFiles, note) { go(Screen.Home) } }
        }
    }
}
