package dev.lizard.receiver

import kotlin.math.roundToInt
import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.HardwareBuffer
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.Process
import android.util.Log
import android.util.Range
import android.util.Size
import android.graphics.SurfaceTexture
import android.view.Surface
import android.view.SurfaceHolder
import android.view.TextureView
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

// ai: The receiver's engine: the native Receiver (Native.kt) and the rear camera (Camera2) feeding it, every change
// ai: on one thread ("lizard-engine") so a start, a stop and a receiver's rebuild never interleave; frames arrive on
// ai: their own ("lizard-frames"). The camera writes two outputs: an ImageReader (for the GPU decoder PRIVATE,
// ai: GPU-sampled, 32 images, each pushed as its HardwareBuffer and closed on the receiver's release; for the C on the
// ai: CPU YUV_420_888, its Y plane pushed and the image closed inside the push) and the TextureView's preview.
class Engine(private val ctx: Context, private val onPhase: (Phase) -> Unit, private val onCamera: (CamInfo) -> Unit) {
    sealed interface Phase {
        data object Idle : Phase
        data object Loading : Phase
        data object Starting : Phase
        data object On : Phase
        data class Error(val why: String) : Phase
    }

    // ai: what the camera was asked for and given, for the Developer readout
    data class CamInfo(val id: String, val format: String, val size: Size, val preview: Size, val fps: Range<Int>,
                       val offered: List<Range<Int>>, val minFrameMs: Double, val sensorOrientation: Int, val note: String)

    // ai: what the rear camera offers, read once at start-up for Settings' choices
    // ai: zoom: the camera's CONTROL_ZOOM_RATIO_RANGE (Android 11 and up), null where it has none (the lens's focus
    // ai: range and calibration, for the Focus slider of 2026-10-04, went with it 2026-10-05)
    data class Caps(val id: String?, val sizes: Set<String>, val lenses: List<Lens>, val zoom: Range<Float>? = null)
    // ai: a back camera an app can open: its id, and a label from its lens (focal length, horizontal field of view,
    // ai: whether it is a logical camera that switches between lenses)
    data class Lens(val id: String, val label: String)

    private val thread = HandlerThread("lizard-engine").apply { start() }
    private val h = Handler(thread.looper)
    private val framesThread = HandlerThread("lizard-frames", Process.THREAD_PRIORITY_DISPLAY).apply { start() }
    private val fh = Handler(framesThread.looper)
    private val main = Handler(Looper.getMainLooper())
    private val cm = ctx.getSystemService(CameraManager::class.java)
    private val images = AtomicLong(0)   // ai: images the reader handed over, for the log's rate beside the camera's
    private val lock = Any()   // ai: guards handle against a destroy while stats(), keep() or clear() reads it
    @Volatile private var handle = 0L
    private var rxKey = ""
    @Volatile private var luma = false   // ai: the receiver reads luma bytes (Native.wantsLuma), so the reader is YUV
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var gen = 0   // ai: bumped by every close: a callback of a camera already closed is ignored
    private val phase = PhaseLock({ Log.i(TAG, it) })   // ai: engine thread only
    // ai: The phase lock learns the display's pace and its gain afresh every run (2026-10-03): from 2026-10-01 to
    // ai: 2026-10-03 they were kept in the preferences (phasePace, phaseGain, no longer read) and given back when track
    // ai: was next set, and a pace learned to its bound (5 ms a second) in 2:1 was given back to every run after it
    // ai: (STATUS "The lock's saved pace").
    @Volatile private var last: android.hardware.camera2.TotalCaptureResult? = null   // ai: the newest result of a frame that was not a delayed one (delay)
    // ai: The newest capture's exposure, sensitivity, frame and readout (the `capture:` line's keys) and the
    // ai: phase lock's snapshot: for the stats row and a replay's meta (MainActivity, 2026-10-04), read from any thread
    data class CameraNow(val exposureMs: Double, val iso: Int, val frameMs: Double, val readoutMs: Double)
    fun cameraNow(): CameraNow? {
        val r = last ?: return null
        return CameraNow((r.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L) / 1e6, r.get(CaptureResult.SENSOR_SENSITIVITY) ?: 0,
            (r.get(CaptureResult.SENSOR_FRAME_DURATION) ?: 0L) / 1e6, (r.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW) ?: 0L) / 1e6)
    }
    fun phaseState(): PhaseLock.State? = phase.state

    // ai: The camera's pipeline slowing (2026-10-07): once
    // ai: its rate a second (its own timestamps) has stood within 5% of the rate asked, a second under 90%
    // ai: of that peak, twice running, is the camera completing fewer frames (the phone warm: the 4K run of 2026-10-01,
    // ai: the 2:1 sessions of 2026-10-06 at 106 to 114 halves a second), and the heat warning says so with the figure
    // ai: (Parts.kt HeatWarning); back at 95% of the peak twice running clears it. Null while nothing is slow; read by
    // ai: Receive on the main thread, set from the capture callback through it.
    data class CameraSlow(val now: Int, val max: Int)
    var cameraSlow by mutableStateOf<CameraSlow?>(null)
        private set
    private fun slowed(v: CameraSlow?, peak: Double) {
        if (v == null) Log.i(TAG, "camera: back at its rate (peak %.1f a second)".format(peak))
        else if (cameraSlow == null) Log.i(TAG, "camera: slowed to %d of %d a second (peak %.1f)".format(v.now, v.max, peak))
        main.post { cameraSlow = v }
    }

    // ai: the preview is a TextureView (a view like any other, so the page can clip it to the crop the decoder reads;
    // ai: a SurfaceView's layer is not clipped by its parents)
    @Volatile private var texture: SurfaceTexture? = null
    // ai: The preview stream's size is the SurfaceTexture's default buffer size at the session's creation
    // ai: (OutputConfiguration(Surface) and the camera service both read it off the surface), and a TextureView sets
    // ai: that size to its own size in pixels whenever it is laid out (TextureView.onSizeChanged, and when it makes
    // ai: the surface). open() set the preview's size and then waited on openCamera; a layout in between (the system
    // ai: bars hiding at Starting, a turn, the crop switched) handed the service the view's size, which it rounds to
    // ai: the nearest size the lens lists, a 4:3 or square one for a tall view, and the view then stretched that
    // ai: picture into its 16:9 box: the preview stretched vertically instead of cropping, on some starts
    // ai: (2026-10-07). Only the preview: the ImageReader's surface has its own fixed size. So the preview's size is
    // ai: kept here, set again right before the session is created (session) and inside every reset by the view
    // ai: (onSurfaceTextureSizeChanged, which the view calls from its own set); once the stream is configured its
    // ai: own dimensions hold and the default no longer matters.
    @Volatile private var previewSize: Size? = null
    val textureListener = object : TextureView.SurfaceTextureListener {
        // ai: a surface that comes while the camera is wanted and closed (the view was remade) reopens it
        override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, hh: Int) {
            texture = st
            h.post { val s = wanted; if (s != null && device == null && !opening) try { open(s) } catch (e: Exception) { fail("The camera could not start: ${e.message}", e) } }
        }
        override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, hh: Int) {
            previewSize?.let { st.setDefaultBufferSize(it.width, it.height) }
        }
        override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean { texture = null; h.post { closeCamera() }; return true }
        override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
    }

    private fun phase(p: Phase) = main.post { onPhase(p) }
    private fun fail(why: String, e: Throwable? = null) {
        Log.e(TAG, why, e)
        closeCamera()
        phase(Phase.Error(why))
    }

    private var wanted: Settings? = null   // ai: the settings the camera was last started with, until it is stopped
    private var opening = false            // ai: an open under way (it waits for the surface itself)
    fun start(s: Settings) = h.post { wanted = s; try { open(s) } catch (e: Exception) { fail("The camera could not start: ${e.message}", e) } }
    // ai: Batch size (Settings.batch), set live on the running receiver and kept for the next one
    fun batch(n: Int) = h.post {
        wanted = wanted?.copy(batch = n.toString())
        val hnd = handle
        if (hnd != 0L) Native.batchCap(hnd, n)
    }
    fun stop() = h.post { wanted = null; closeCamera(); phase(Phase.Idle) }
    fun shutdown() = h.post {
        closeCamera(); destroyReceiver()
        thread.quitSafely(); framesThread.quitSafely()
    }

    fun stats(): String = synchronized(lock) { if (handle != 0L) Native.stats(handle) else "" }
    // ai: the transfer in hand forgotten (2026-10-07, the received file deleted on Home): the receiver no longer reports
    // ai: the file, so the state line's "Received" goes, and the same file in the light is received and kept anew
    // ai: The receiver forgets its transfer, ids and held word (Native.clear), on the caller's thread and done when it
    // ai: returns, so a stats read after it never shows what it forgot (2026-10-10: posted to the engine's thread until
    // ai: then, and a poll in between filed the forgotten file again, MainActivity.forgetReceived). Never on the main
    // ai: thread: it waits for the transfer's worker (XferRx::clear).
    fun clear() = synchronized(lock) { if (handle != 0L) Native.clear(handle) }
    // ai: Save replays (2026-10-03, MainActivity's replay): the replay (Native.replayNew) handed to the receiver, and to
    // ai: one made while it runs, which hands it each frame it decodes; replayEnd takes replay r back where it is the
    // ai: one handed over, then ends its run on a thread of its own (the frames the decoder still holds for it first,
    // ai: then the renames and the meta), done on the main thread with its JSON (Native.replayEnd) or the failure. Its
    // ai: run ends even where this engine has shut down (its thread gone: the activity's onDestroy ends a run, then
    // ai: shuts it down), since the receiver went with it.
    private var replaying = 0L   // ai: engine thread only
    fun replay(r: Long) = h.post { replaying = r; synchronized(lock) { if (handle != 0L) Native.record(handle, r) } }
    fun replayEnd(r: Long, rxStats: String, rows: String, more: String, done: (Result<String>) -> Unit) {
        val end = {
            if (r != 0L) Thread({ val res = runCatching { Native.replayEnd(r, rxStats, rows, more) }; main.post { done(res) } }, "lizard-replay").start()
            else main.post { done(Result.failure(IllegalStateException("no replay"))) }
        }
        if (!h.post {
            if (replaying == r) { replaying = 0L; synchronized(lock) { if (handle != 0L) Native.record(handle, 0L) } }
            end()
        }) end()
    }
    // ai: The verified file moved into the received files (Library.dest, 2026-10-01): on this thread, so it never races
    // ai: a receiver's rebuild (which empties the store it lies in); a rename, the two in one filesystem. done(true) on
    // ai: the main thread once it is there. Native.file's path is stale after it: nothing reads the store's copy again.
    fun keep(dest: File, done: (Boolean) -> Unit) = h.post {
        val path = synchronized(lock) { if (handle != 0L) Native.file(handle) else "" }
        val ok = path.isNotEmpty() && File(path).renameTo(dest)
        if (!ok) Log.w(TAG, "keep: $path to $dest failed")
        main.post { done(ok) }
    }

    fun lenses(): List<Lens> = backIds().map { id ->
        val ch = cm.getCameraCharacteristics(id)
        val f = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
        val sw = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.width
        val fov = if (f != null && sw != null && f > 0) Math.toDegrees(2 * Math.atan(sw / (2.0 * f))).roundToInt() else null
        val logical = ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            ?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA) == true
        Lens(id, listOfNotNull("$id", f?.let { "%.1f mm".format(it) }, fov?.let { "$it°" }, if (logical) "multi" else null).joinToString(" "))
    }.also { Log.i(TAG, "back cameras: ${it.joinToString("; ") { l -> l.label }}") }

    fun caps(s: Settings): Caps {
        val id = rearId(s.camera) ?: return Caps(null, emptySet(), emptyList())
        val ch = cm.getCameraCharacteristics(id)
        val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val all = (sizes(map, ImageFormat.PRIVATE) + sizes(map, ImageFormat.YUV_420_888))
            .map { "${it.width}x${it.height}" }.toSet()
        val zr = if (Build.VERSION.SDK_INT >= 30) ch.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE) else null
        return Caps(id, Settings.RESOLUTIONS.filter { it in all }.toSet(), lenses(), zr)
    }

    private fun backIds() = cm.cameraIdList.filter {
        cm.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
    }
    // ai: The camera Settings names (Settings.camera); else (none chosen, or one no longer listed) the back camera that focuses
    // ai: closest (the largest LENS_INFO_MINIMUM_FOCUS_DISTANCE, in dioptres), the first listed among equals: a receiver
    // ai: is held near a screen. On the S26 that is the 2.2 mm camera (5 cm), where the first listed, the main lens (10
    // ai: cm, a logical multi-camera), read 38 KB/s of a stream the 2.2 mm read 363 of (2026-09-30). Public since
    // ai: 2026-10-04: the lens's own settings are kept under this id (Settings.forCamera).
    fun rearId(camera: String): String? {
        val ids = backIds()
        ids.firstOrNull { it == camera }?.let { return it }
        return ids.maxByOrNull { cm.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f }
    }

    private fun sizes(map: android.hardware.camera2.params.StreamConfigurationMap, fmt: Int) =
        map.getOutputSizes(fmt)?.toList().orEmpty()

    private fun ranges(ch: CameraCharacteristics) =
        ch.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.toList().orEmpty()

    // ai: [60,60] where offered, else the highest fixed range (a fixed range holds the exposure time to the frame), else
    // ai: the highest top. A range chosen by hand went 2026-10-01: the receiver and the lock run at 60.
    private fun pickFps(offered: List<Range<Int>>): Range<Int> {
        offered.firstOrNull { it.lower == 60 && it.upper == 60 }?.let { return it }
        return offered.filter { it.lower == it.upper }.maxByOrNull { it.upper } ?: offered.maxBy { it.upper }
    }

    private fun ensureReceiver(s: Settings) {
        val key = "${s.decoder}/${s.precision}/${s.layout}"
        if (handle != 0L && key == rxKey) return
        destroyReceiver()
        phase(Phase.Loading)
        val assets = Assets.ensure(ctx)
        val cache = File(ctx.cacheDir, "lizard").apply { mkdirs() }
        val store = File(ctx.filesDir, "received").apply { mkdirs() }
        val hnd = Native.create(assets.path, cache.path, store.path, s.decoder, s.precision, s.layout)
        luma = Native.wantsLuma(hnd)
        Native.batchCap(hnd, (wanted ?: s).frames)
        if (replaying != 0L) Native.record(hnd, replaying)
        synchronized(lock) { handle = hnd }
        rxKey = key
    }

    private fun destroyReceiver() {
        val hnd = handle
        if (hnd == 0L) return
        synchronized(lock) { handle = 0L }
        Native.destroy(hnd)
        Native.closeAll(this)   // ai: a receiver's destructor releases what it holds; anything of this engine's left is closed here
    }

    private var zoom = 1f
    // ai: The running session's zoom (Settings' slider, 2026-10-01): the repeating
    // ai: request issued again with the new ratio, the camera kept open (a restart cost half a second and the phase
    // ai: lock's footing). Null with no session. A drag posts many values: the engine thread takes the newest once.
    private var rezoom: ((Float) -> Unit)? = null
    @Volatile private var zoomAsked = 1f
    private val zoomPosted = java.util.concurrent.atomic.AtomicBoolean(false)
    fun zoom(z: Float) {
        zoomAsked = z
        if (zoomPosted.compareAndSet(false, true)) h.post {
            zoomPosted.set(false)
            val v = zoomAsked
            zoom = v
            wanted = wanted?.copy(zoom = "%.1f".format(java.util.Locale.ROOT, v))   // ai: a camera opened again keeps it
            rezoom?.invoke(v)
        }
    }

    // ai: Autofocus: the camera's continuous video mode where it lists one (a lens held in dioptres, Receive's Focus of
    // ai: 2026-10-04, went 2026-10-05)
    private fun applyFocus(b: CaptureRequest.Builder, ch: CameraCharacteristics) {
        val af = ch.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)?.toList().orEmpty()
        if (CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO in af) b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
    }

    @SuppressLint("MissingPermission")   // ai: MainActivity starts the engine only once the permission is granted
    private fun open(s: Settings) {
        zoom = s.zoom.toFloatOrNull() ?: 1f
        closeCamera()
        opening = true
        ensureReceiver(s)
        phase(Phase.Starting)
        val id = rearId(s.camera) ?: return fail("This phone has no rear camera.")
        val ch = cm.getCameraCharacteristics(id)
        val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val (ww, wh) = s.resolution.split("x").map { it.toInt() }
        val want = Size(ww, wh)
        val priv = sizes(map, ImageFormat.PRIVATE)
        val yuv = sizes(map, ImageFormat.YUV_420_888)
        var note = ""
        val size = if (want in priv || want in yuv) want else {
            val near = (priv.ifEmpty { yuv }).minBy { Math.abs(it.width * it.height - ww * wh) }
            note = "no $want output, $near instead"
            near
        }
        val offered = ranges(ch)
        val fps = pickFps(offered)
        // ai: the preview: the largest of the same shape up to 1280 wide (it is only for aiming), else the capture's size
        val preview = map.getOutputSizes(SurfaceHolder::class.java)?.toList().orEmpty()
            .filter { it.width * size.height == it.height * size.width && it.width <= 1280 }
            .maxByOrNull { it.width } ?: size
        val g = gen
        // ai: the session needs the preview's texture: wait (2 s at most) for the view to make it
        val t0 = System.nanoTime()
        while (texture == null && System.nanoTime() - t0 < 2_000_000_000L) Thread.sleep(10)
        val st = texture ?: return fail("The camera preview did not appear.")
        previewSize = preview
        st.setDefaultBufferSize(preview.width, preview.height)
        val surface = Surface(st)
        Log.i(TAG, "open camera $id: $size, preview $preview, fps $fps")
        cm.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(d: CameraDevice) {
                if (g != gen) { d.close(); return }
                device = d
                opening = false
                // ai: the CPU decoder reads the Y plane's bytes: YUV_420_888 for it, the GPU's own layout otherwise
                val fmt = if (size in priv && !luma) ImageFormat.PRIVATE else ImageFormat.YUV_420_888
                session(d, g, fmt, size, surface, fps, ch) { format ->
                    val minNs = map.getOutputMinFrameDuration(format, size)
                    main.post {
                        onCamera(CamInfo(id, if (format == ImageFormat.PRIVATE) "PRIVATE" else "YUV_420_888", size,
                            preview, fps, offered, minNs / 1e6, ch.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0, note))
                    }
                }
            }
            override fun onDisconnected(d: CameraDevice) { if (g == gen) fail("The camera was taken by another app.") else d.close() }
            override fun onError(d: CameraDevice, error: Int) {
                if (g == gen) fail(if (error == ERROR_CAMERA_IN_USE || error == ERROR_MAX_CAMERAS_IN_USE)
                    "The camera is in use by another app." else "The camera stopped (error $error).") else d.close()
            }
        }, h)
    }

    // ai: PRIVATE first (the GPU's own layout, no CPU copy); YUV_420_888 with the same GPU usage (and CPU reads, so a
    // ai: CPU decoder can lock its Y plane) where the session refuses it
    private fun session(d: CameraDevice, g: Int, fmt: Int, size: Size, preview: android.view.Surface, fps: Range<Int>,
                        ch: CameraCharacteristics, ok: (Int) -> Unit) {
        val usage = if (luma) HardwareBuffer.USAGE_CPU_READ_OFTEN else HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE or
            (if (fmt == ImageFormat.PRIVATE) 0L else HardwareBuffer.USAGE_CPU_READ_OFTEN)
        val r = try { ImageReader.newInstance(size.width, size.height, fmt, MAX_IMAGES, usage) } catch (e: Exception) {
            if (fmt == ImageFormat.PRIVATE) return session(d, g, ImageFormat.YUV_420_888, size, preview, fps, ch, ok)
            return fail("The camera's frames cannot be read: ${e.message}", e)
        }
        reader = r
        r.setOnImageAvailableListener({ onImage(it) }, fh)
        // ai: the preview's size set again here, on the thread that creates the session and right before it does:
        // ai: the view may have reset it since open() (previewSize above)
        previewSize?.let { p -> texture?.setDefaultBufferSize(p.width, p.height) }
        val outs = listOf(OutputConfiguration(r.surface), OutputConfiguration(preview))
        val exec = Executor { h.post(it) }
        // ai: the call itself can throw (2026-10-06, the S26: CameraAccessException "Error configuring streams: Broken
        // ai: pipe" from endConfigure, the camera service's side gone, which killed the app on the engine thread):
        // ai: a failure the state line reports, and Start tries again
        try { d.createCaptureSession(SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outs, exec,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(cs: CameraCaptureSession) {
                    if (g != gen) { cs.close(); return }
                    session = cs
                    val b = d.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
                    b.addTarget(r.surface); b.addTarget(preview)
                    // ai: The ISP's enhancements off, each where the lens lists its off mode: the camera is a machine
                    // ai: vision camera here, and every filter that makes a picture for a person is a filter over the
                    // ai: code the decoder then reads through. Noise reduction since 2026-09-30; since 2026-10-06 edge
                    // ai: enhancement (sharpening: an overshoot on every module edge), hot pixel correction, chromatic
                    // ai: aberration correction, face detection and scene modes (the S26 keeps scene mode 1 whatever is
                    // ai: asked) and the statistics maps. Measured for edge and hot pixel in 2:1 (two LIZARD-480 painted
                    // ai: 60, the S26 by hand, the same minutes): 51.7 blocks of 60 a clean capture and 2.8 MB/s off
                    // ai: against 34.5 to 39.5 and 1.6 to 2.2 MB/s with the record template's FAST (STATUS "The ISP's
                    // ai: enhancements off"). Three stay the camera's, since all three off together read worse by
                    // ai: hand the same evening (which of them is not known): its tone curve (the sRGB preset, or
                    // ai: an sRGB contrast curve, is the off form here), lens shading correction and distortion
                    // ai: correction. The `capture:` line reports what the camera did. test: `adb shell setprop
                    // ai: debug.lizard.camx <name,...>` (read at open): a name leaves that one as the template has it
                    // ai: (nr, edge, hot, ca, face, scene, stats; `all` every one), `-tone`, `-shade`, `-dist` ask one of
                    // ai: the three off: the A/B's arms either way.
                    val lists = { key: CameraCharacteristics.Key<IntArray> -> ch.get(key)?.toList().orEmpty() }
                    val camxAsked = Native.prop("debug.lizard.camx").split(',').map { it.trim() }.filter { it.isNotEmpty() }
                    val keep = setOf("tone", "shade", "dist") + camxAsked.filter { !it.startsWith("-") } - camxAsked.filter { it.startsWith("-") }.map { it.drop(1) }.toSet()
                    val off = { name: String -> !("all" in keep || "0" in keep || name in keep) }
                    if (off("nr") && CameraMetadata.NOISE_REDUCTION_MODE_OFF in lists(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES))
                        b.set(CaptureRequest.NOISE_REDUCTION_MODE, CameraMetadata.NOISE_REDUCTION_MODE_OFF)
                    if (off("edge") && CameraMetadata.EDGE_MODE_OFF in lists(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES)) b.set(CaptureRequest.EDGE_MODE, CameraMetadata.EDGE_MODE_OFF)
                    if (off("hot") && CameraMetadata.HOT_PIXEL_MODE_OFF in lists(CameraCharacteristics.HOT_PIXEL_AVAILABLE_HOT_PIXEL_MODES)) b.set(CaptureRequest.HOT_PIXEL_MODE, CameraMetadata.HOT_PIXEL_MODE_OFF)
                    if (off("ca") && CameraMetadata.COLOR_CORRECTION_ABERRATION_MODE_OFF in lists(CameraCharacteristics.COLOR_CORRECTION_AVAILABLE_ABERRATION_MODES))
                        b.set(CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE, CameraMetadata.COLOR_CORRECTION_ABERRATION_MODE_OFF)
                    if (off("face") && CameraMetadata.STATISTICS_FACE_DETECT_MODE_OFF in lists(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_FACE_DETECT_MODES))
                        b.set(CaptureRequest.STATISTICS_FACE_DETECT_MODE, CameraMetadata.STATISTICS_FACE_DETECT_MODE_OFF)
                    if (off("scene") && CameraMetadata.CONTROL_SCENE_MODE_DISABLED in lists(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES)) {
                        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO); b.set(CaptureRequest.CONTROL_SCENE_MODE, CameraMetadata.CONTROL_SCENE_MODE_DISABLED)
                    }
                    if (off("stats")) {
                        if (CameraMetadata.STATISTICS_LENS_SHADING_MAP_MODE_OFF in lists(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_LENS_SHADING_MAP_MODES))
                            b.set(CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE, CameraMetadata.STATISTICS_LENS_SHADING_MAP_MODE_OFF)
                        if (ch.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_HOT_PIXEL_MAP_MODES)?.contains(false) == true) b.set(CaptureRequest.STATISTICS_HOT_PIXEL_MAP_MODE, false)
                        if (Build.VERSION.SDK_INT >= 28 && CameraMetadata.STATISTICS_OIS_DATA_MODE_OFF in lists(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_OIS_DATA_MODES))
                            b.set(CaptureRequest.STATISTICS_OIS_DATA_MODE, CameraMetadata.STATISTICS_OIS_DATA_MODE_OFF)
                    }
                    if (off("tone")) {
                        val modes = lists(CameraCharacteristics.TONEMAP_AVAILABLE_TONE_MAP_MODES)
                        if (CameraMetadata.TONEMAP_MODE_PRESET_CURVE in modes) {
                            b.set(CaptureRequest.TONEMAP_MODE, CameraMetadata.TONEMAP_MODE_PRESET_CURVE); b.set(CaptureRequest.TONEMAP_PRESET_CURVE, CameraMetadata.TONEMAP_PRESET_CURVE_SRGB)
                        } else if (CameraMetadata.TONEMAP_MODE_CONTRAST_CURVE in modes) {
                            val n = minOf(ch.get(CameraCharacteristics.TONEMAP_MAX_CURVE_POINTS) ?: 2, 64).coerceAtLeast(2)
                            val pts = FloatArray(2 * n) { i -> val x = (i / 2) / (n - 1f); if (i % 2 == 0) x else if (x <= 0.0031308f) 12.92f * x else 1.055f * Math.pow(x.toDouble(), 1 / 2.4).toFloat() - 0.055f }
                            b.set(CaptureRequest.TONEMAP_MODE, CameraMetadata.TONEMAP_MODE_CONTRAST_CURVE); b.set(CaptureRequest.TONEMAP_CURVE, android.hardware.camera2.params.TonemapCurve(pts, pts, pts))
                        }
                    }
                    if (off("shade") && CameraMetadata.SHADING_MODE_OFF in lists(CameraCharacteristics.SHADING_AVAILABLE_MODES)) b.set(CaptureRequest.SHADING_MODE, CameraMetadata.SHADING_MODE_OFF)
                    if (off("dist") && Build.VERSION.SDK_INT >= 28 && CameraMetadata.DISTORTION_CORRECTION_MODE_OFF in lists(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES))
                        b.set(CaptureRequest.DISTORTION_CORRECTION_MODE, CameraMetadata.DISTORTION_CORRECTION_MODE_OFF)
                    if (camxAsked.isNotEmpty()) Log.i(TAG, "camx: ${camxAsked.joinToString(",")} (kept as the template has them: ${keep.joinToString(",")})")
                    // ai: test: `adb shell setprop debug.lizard.camdump 1` logs every key of the request as built, the
                    // ai: vendor's included, one `request:` line a key at this open: what the template turns on that the
                    // ai: `capture:` line cannot name
                    if (Native.prop("debug.lizard.camdump") == "1") { val q = b.build(); for (k in q.keys) Log.i(TAG, "request: ${k.name} = ${q.get(k).let { v -> if (v is IntArray) v.toList().toString() else if (v is FloatArray) v.toList().toString() else v.toString() }}") }
                    // ai: The zoom (Settings.zoom): a crop of the sensor, so the code sits in the middle of the lens's
                    // ai: field and the phone further from its close-focus limit. S26, the 2.2 mm camera, LIZARD-512
                    // ai: aimed by hand to fill the square at each zoom (2026-09-30): 1.0x 398 KB/s, 1.2x 456, 1.4x 589
                    // ai: and 509, 1.7x 517, 2.8x 352; video stabilisation (a crop too, which is how Chrome's camera
                    // ai: came out ahead) 499 to 581.
                    val zr = if (Build.VERSION.SDK_INT >= 30) ch.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE) else null
                    if (zr != null) b.set(CaptureRequest.CONTROL_ZOOM_RATIO, zoom.coerceIn(zr.lower, zr.upper))
                    b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
                    applyFocus(b, ch)
                    // ai: stabilisation off: it warps and crops each frame differently, which the border would have to chase
                    val vs = ch.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)?.toList().orEmpty()
                    if (CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF in vs)
                        b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                    val ois = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)?.toList().orEmpty()
                    if (CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF in ois)
                        b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)
                    // ai: what the camera did, every five seconds, for the log (the request only asks); and the delay
                    // ai: test: `adb shell setprop debug.lizard.nudge <us>[.<anything>]` (read once a second; any change
                    // ai: of the value asks again) makes one frame that much longer and logs the capture intervals
                    // ai: around it, the delayed frame's marked *
                    val told = object : CameraCaptureSession.CaptureCallback() {
                        var n = 0
                        var t = 0L
                        var seen = 0L
                        var prevTs = 0L
                        var watch = 0
                        var secAt = 0L; var secN = 0; var peak = 0.0; var slowFor = 0; var okFor = 0   // ai: the rate a second, for cameraSlow
                        var asked = Native.prop("debug.lizard.nudge")
                        val gaps = StringBuilder()
                        // ai: test: `adb shell setprop debug.lizard.resolution <WxH>` reopens the camera at that size
                        // ai: as the Developer panel's Resolution does (the receiver kept, its plan grown if the crop
                        // ai: is larger: STATUS "Higher resolutions in the Android app"); acted on when the value
                        // ai: changes, so the panel's own choice stands until the property is set again
                        var resAsked = Native.prop("debug.lizard.resolution")
                        // ai: the camera's phase against the display (PhaseLock.kt, the engine's one: what it has
                        // ai: learned of the two clocks outlives a camera opened again), by `adb shell setprop
                        // ai: debug.lizard.phase <us> | scan | track` (read twice a second): a hold at that
                        // ai: point of a 60 Hz grid, a scan of every point with a line each, or track's automatic
                        // ai: hold; unset, track (always on since 2026-10-10: the Phase lock setting went). Anything else: off.
                        init { phase.reopened() }
                        override fun onCaptureCompleted(s: CameraCaptureSession, q: CaptureRequest, r: android.hardware.camera2.TotalCaptureResult) {
                            val ts = r.get(android.hardware.camera2.CaptureResult.SENSOR_TIMESTAMP) ?: 0L
                            val delayed = q.get(CaptureRequest.CONTROL_AE_MODE) == CameraMetadata.CONTROL_AE_MODE_OFF
                            // ai: a delay is asked from the camera's own frame: a delayed frame's duration is not it
                            if (!delayed) last = r
                            if (watch > 0 && prevTs != 0L) {
                                gaps.append(" %.3f".format((ts - prevTs) / 1e6))
                                if (delayed) gaps.append("*(%.3f)".format((r.get(android.hardware.camera2.CaptureResult.SENSOR_FRAME_DURATION) ?: 0L) / 1e6))
                                if (--watch == 0) { Log.i(TAG, "delay: intervals ms$gaps"); gaps.setLength(0) }
                            }
                            prevTs = ts
                            // ai: the camera's rate a second against its peak (cameraSlow above)
                            if (secAt == 0L) secAt = ts
                            secN++
                            if (ts - secAt >= 1_000_000_000L) {
                                val rate = secN * 1e9 / (ts - secAt); secAt = ts; secN = 0
                                if (rate > peak) peak = rate
                                val max = fps.upper
                                val slow = cameraSlow
                                if (peak >= 0.95 * max && rate < 0.9 * peak) { okFor = 0; if (++slowFor >= 2 && slow?.now != rate.roundToInt()) slowed(CameraSlow(rate.roundToInt(), max), peak) }
                                else { slowFor = 0; if (slow != null && rate >= 0.95 * peak && ++okFor >= 2) slowed(null, peak) }
                            }
                            if (n % 30 == 15) {
                                val p = Native.prop("debug.lizard.nudge")
                                if (p != asked) { asked = p; p.substringBefore('.').toLongOrNull()?.let { us -> if (us > 0) { watch = 30; delay(cs, b, us, this) } } }
                                val res = Native.prop("debug.lizard.resolution")
                                if (res != resAsked) {
                                    resAsked = res
                                    val w = wanted
                                    if (w != null && res.matches(Regex("\\d+x\\d+")) && res != w.resolution) { Log.i(TAG, "resolution: $res asked"); start(w.copy(resolution = res)) }
                                }
                                // ai: `debug.lizard.pilots`: 0, track reads the frames by their blocks even where they
                                // ai: flash (the A/B's other arm); unset or anything else, where they flash it holds
                                // ai: by the pilots' leaks (PhaseLock.kt)
                                phase.usePilots = Native.prop("debug.lizard.pilots") != "0"
                                // ai: track always (2026-10-10: the setting and its off went); `debug.lizard.phase` the tools' override
                                // ai: (scan, track, off, or a hold at that many us)
                                when (val want = Native.prop("debug.lizard.phase").ifEmpty { "track" }) {
                                    "scan" -> phase.set(PhaseLock.Mode.Scan)
                                    "track" -> phase.set(PhaseLock.Mode.Track)
                                    else -> want.toLongOrNull()?.let { phase.set(PhaseLock.Mode.Hold, it) } ?: phase.set(PhaseLock.Mode.Off)
                                }
                            }
                            // ai: the lock's frames as the receiver reads them, at every capture (twice a second
                            // ai: from the stats' JSON to 2026-10-01: a reading reached the lock 0.7 s old)
                            feed(phase)
                            val us = phase.onCapture(ts, delayed)
                            if (us > 0) delay(cs, b, us, this, quiet = true)
                            if (n % 60 == 0) rx()?.let { Log.i(TAG, it) }
                            if (n % 300 == 0) phase.line(ts)?.let { Log.i(TAG, it) }
                            if (n++ % 300 != 0) return
                            val now = System.nanoTime()
                            val im = images.get()
                            if (t != 0L) Log.i(TAG, "camera: %.1f captures a second, %.1f images, %d held".format(300e9 / (now - t), (im - seen) * 1e9 / (now - t), Native.heldBy(this@Engine)))
                            t = now; seen = im
                            // ai: `readout`: the sensor's first row to its last (the rolling shutter), which with the
                            // ai: exposure is how long a capture looks at the display
                            Log.i(TAG, "capture: exposure %.2f ms, iso %d, frame %.2f ms, readout %.2f ms, focus %.2f D (af state %d), eis %d, ois %d, nr %d, edge %d, face %d, scene %d, hot %d, tone %d, shade %d, dist %d, zoom %.2f".format(
                                (r.get(android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L) / 1e6, r.get(android.hardware.camera2.CaptureResult.SENSOR_SENSITIVITY) ?: 0,
                                (r.get(android.hardware.camera2.CaptureResult.SENSOR_FRAME_DURATION) ?: 0L) / 1e6, (r.get(android.hardware.camera2.CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW) ?: 0L) / 1e6,
                                r.get(android.hardware.camera2.CaptureResult.LENS_FOCUS_DISTANCE) ?: 0f,
                                r.get(android.hardware.camera2.CaptureResult.CONTROL_AF_STATE) ?: -1, r.get(android.hardware.camera2.CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE) ?: -1,
                                r.get(android.hardware.camera2.CaptureResult.LENS_OPTICAL_STABILIZATION_MODE) ?: -1, r.get(android.hardware.camera2.CaptureResult.NOISE_REDUCTION_MODE) ?: -1,
                                r.get(android.hardware.camera2.CaptureResult.EDGE_MODE) ?: -1,
                                r.get(android.hardware.camera2.CaptureResult.STATISTICS_FACE_DETECT_MODE) ?: -1, r.get(android.hardware.camera2.CaptureResult.CONTROL_SCENE_MODE) ?: -1,
                                r.get(android.hardware.camera2.CaptureResult.HOT_PIXEL_MODE) ?: -1,
                                r.get(android.hardware.camera2.CaptureResult.TONEMAP_MODE) ?: -1, r.get(android.hardware.camera2.CaptureResult.SHADING_MODE) ?: -1,
                                (if (Build.VERSION.SDK_INT >= 28) r.get(android.hardware.camera2.CaptureResult.DISTORTION_CORRECTION_MODE) else null) ?: -1,
                                if (Build.VERSION.SDK_INT >= 30) r.get(android.hardware.camera2.CaptureResult.CONTROL_ZOOM_RATIO) ?: 1f else 1f))
                        }
                    }
                    try { cs.setRepeatingRequest(b.build(), told, h) } catch (e: Exception) { return fail("The camera stopped: ${e.message}", e) }
                    rezoom = if (zr == null) null else { z ->
                        if (g == gen) {
                            b.set(CaptureRequest.CONTROL_ZOOM_RATIO, z.coerceIn(zr.lower, zr.upper))
                            try { cs.setRepeatingRequest(b.build(), told, h) } catch (e: Exception) { Log.w(TAG, "zoom: ${e.message}") }
                        }
                    }
                    ok(fmt)
                    phase(Phase.On)
                }
                override fun onConfigureFailed(cs: CameraCaptureSession) {
                    if (g != gen) return
                    closeReader()
                    if (fmt == ImageFormat.PRIVATE) {
                        Log.w(TAG, "PRIVATE ${size} refused, YUV_420_888 instead")
                        session(d, g, ImageFormat.YUV_420_888, size, preview, fps, ch, ok)
                    } else fail("The camera refused ${size.width}x${size.height}.")
                }
            })) } catch (e: Exception) { closeReader(); fail("The camera could not start: ${e.message}", e) }
    }

    // ai: A second's line for the log while a symbol is in view: what the receiver read (the rig's stats rows say the
    // ai: same when the rig is up; this needs only adb), and what it could have read at this format (2026-09-30). With
    // ai: the phase held every capture is a clean one, so the ceiling is a clean capture's blocks at the painted rate;
    // ai: it is read off the captures since the last line, whatever share of them the phase let through: clean are
    // ai: those with at least half the blocks
    // ai: of the third best of them (and an eighth of the word's), and the ceiling is their mean, 469 B a block, at
    // ai: the painted rate or the camera's if that is lower. tools/phone/ladder.py makes the table by format.
    private var rxSeen = 0.0   // ai: the newest capture already counted, ms on the camera's clock (engine thread)
    private fun rx(): String? {
        val j = try { org.json.JSONObject(stats()) } catch (_: Exception) { return null }
        val a = j.optJSONArray("series")
        val blocks = ArrayList<Int>()
        var newest = rxSeen; var first = Double.MAX_VALUE; var all = 0
        if (a != null) for (i in 0 until a.length()) {
            val f = a.optJSONArray(i) ?: continue
            val ms = f.optDouble(0)
            if (ms <= rxSeen) continue
            all++
            if (ms > newest) newest = ms
            if (ms < first) first = ms
            if (f.optInt(3) == 1) blocks.add(f.optInt(1))
        }
        rxSeen = newest
        val w = j.optJSONObject("word")
        val version = w?.optInt("version") ?: 0
        val painted = w?.optInt("fps") ?: 0
        if (version == 0 || blocks.size < 8 || 2 * blocks.size < all) return null
        blocks.sortDescending()
        val least = maxOf(1, (blocks[minOf(2, blocks.size - 1)] + 1) / 2, (Native.blocksFor(8 * version) + 7) / 8)
        val clean = blocks.filter { it >= least }
        val each = if (clean.isEmpty()) 0.0 else clean.average()
        val camera = if (all > 1 && newest > first) (all - 1) * 1000.0 / (newest - first) else 0.0
        val rate = if (painted > 0) minOf(painted.toDouble(), camera) else camera
        // ai: the symbol's side in camera px (the receiver's mean over the frames it found) against the crop each frame
        // ai: gives it (2:1: half the frame's long side, at most its short side), and the share of frames found
        // ai: (2026-10-03, to tell a code that is far from one that sits off its half: the 2.8 MB/s run read 50.8
        // ai: of 52 blocks a clean capture, later runs 38 to 44 with timing, heat and leaks the same)
        val crop = wanted?.resolution?.split("x")?.mapNotNull { it.toIntOrNull() }?.takeIf { it.size == 2 }?.let { (w, h) ->
            if (wanted?.layout == "2:1") minOf(maxOf(w, h) / 2, minOf(w, h)) else minOf(w, h) } ?: 0
        return "rx: %s, LIZARD-%d painted %d, goodput %.0f KB/s, %d of %d captures clean, %.1f blocks a clean capture, ceiling %.0f KB/s, %d frames a second, symbol %.0f px of a %d crop, found %.0f%%".format(
            j.optString("state"), 8 * version, painted, j.optDouble("goodputKBs", 0.0), clean.size, blocks.size, each, each * 469 * rate / 1000, j.optInt("processedFps"),
            j.optDouble("side", 0.0), crop, 100 * j.optDouble("foundShare", 0.0))
    }

    // ai: The receiver's frames since the last call (Native.series: ms on the camera's clock, verified blocks, new
    // ai: blocks, registered, the pilots' r and its standard error over the even blocks, then over the odd, NaN
    // ai: where none) and the word's blocks a frame, to the phase's scan or track (each frame's a launch after its
    // ai: capture: since 2026-10-08 the GPU receiver launches as soon as a frame is staged and a lane is free, never
    // ai: waiting to fill a batch). Called at
    // ai: every capture result on the camera's thread since 2026-10-01: one JNI call and no JSON, so a call with
    // ai: nothing new costs microseconds.
    private var fedMs = -1.0     // ai: the newest frame already fed, ms on the camera's clock
    private fun feed(p: PhaseLock) {
        val hnd = handle
        if (hnd == 0L) return
        if (p.mode != PhaseLock.Mode.Scan && p.mode != PhaseLock.Mode.Track) return
        val t0 = System.nanoTime()
        val a = try { Native.series(hnd, fedMs) } catch (_: Throwable) { return }
        val n = (a.size - 1) / 8
        if (n <= 0) return
        val frames = ArrayList<PhaseLock.Frame>(n)
        for (i in 0 until n) { val o = 1 + 8 * i; frames.add(PhaseLock.Frame((a[o] * 1e6).toLong(), a[o + 1].toInt(), a[o + 3] == 1.0, a[o + 4], a[o + 5], a[o + 6], a[o + 7])) }
        fedMs = a[1 + 8 * (n - 1)]
        p.onWindow(frames, a[0].toInt())
        // ai: this runs on the camera's thread between capture results: a slow feed delays the lock's next delay
        // ai: (2026-09-30: the search's steps grew to 2.4 to 3.9 ms on the S26, one step every 16 to 26 captures)
        val t1 = System.nanoTime()
        if (t1 - t0 > 20_000_000L) Log.i(TAG, "phase: the feed took %d ms".format((t1 - t0) / 1_000_000))
    }

    // ai: One frame made `us` microseconds longer, so every frame after it starts that much later: the camera's phase
    // ai: against the display's refresh, moved (2026-09-30; a browser's camera has no such control). Camera2 takes a
    // ai: frame's duration from the request only with auto-exposure off, so the one frame carries the sensor's last
    // ai: exposure and sensitivity itself; the repeating request, with auto-exposure, goes on after it.
    private fun delay(cs: CameraCaptureSession, b: CaptureRequest.Builder, us: Long, cb: CameraCaptureSession.CaptureCallback, quiet: Boolean = false) {
        val r = last ?: return
        val exp = r.get(android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME) ?: return
        val iso = r.get(android.hardware.camera2.CaptureResult.SENSOR_SENSITIVITY) ?: return
        val dur = r.get(android.hardware.camera2.CaptureResult.SENSOR_FRAME_DURATION) ?: return
        b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exp)
        b.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
        b.set(CaptureRequest.SENSOR_FRAME_DURATION, dur + us * 1000)
        val one = b.build()
        b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
        b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, null)
        b.set(CaptureRequest.SENSOR_SENSITIVITY, null)
        b.set(CaptureRequest.SENSOR_FRAME_DURATION, null)
        try {
            cs.capture(one, cb, h)
            if (!quiet) Log.i(TAG, "delay: one frame of %.3f ms asked (%.3f and %.3f more), exposure %.2f ms, iso %d".format((dur + us * 1000) / 1e6, dur / 1e6, us / 1e3, exp / 1e6, iso))
        } catch (e: Exception) { Log.w(TAG, "delay: ${e.message}") }
    }

    // ai: every image the reader holds, in order; one the receiver cannot take is released by it at once, and one push
    // ai: refuses (no buffer, no receiver) is closed here
    private fun onImage(r: ImageReader) {
        while (true) {
            val img = try { r.acquireNextImage() } catch (_: IllegalStateException) { null } ?: return
            images.incrementAndGet()
            val hnd = handle
            if (hnd == 0L) { img.close(); continue }
            // ai: the CPU decoder takes plane 0 (Y), copies its crop inside push and releases the image there
            val hb = if (luma) null else img.hardwareBuffer
            val plane = if (hb == null) img.planes.firstOrNull() else null
            val tag = Native.hold(img, hb, this)
            try {
                Native.push(hnd, hb, plane?.buffer, img.width, img.height, plane?.rowStride ?: img.width, img.timestamp, tag)
            } catch (e: IllegalArgumentException) {
                Log.w(TAG, "push: ${e.message}")
                Native.release(tag)
            }
        }
    }

    private fun closeReader() {
        val r = reader ?: return
        reader = null
        r.setOnImageAvailableListener(null, null)
        // ai: no push in flight past this point (the frames thread drained), then a second for the receiver to hand
        // ai: back what it holds before the reader goes (a HardwareBuffer still held keeps its memory alive regardless)
        val done = CountDownLatch(1)
        fh.post { done.countDown() }
        done.await(1, TimeUnit.SECONDS)
        val t0 = System.nanoTime()
        while (Native.heldBy(this) > 0 && System.nanoTime() - t0 < 1_000_000_000L) Thread.sleep(5)
        Native.heldBy(this).let { if (it > 0) Log.w(TAG, "$it frames still held as the reader closes") }
        // ai: the receiver drops its imports of this reader's buffers (2026-10-02): kept across a stop and start, it
        // ai: imported each new reader's buffers beside the old, and the third camera opening in one receiver (a file,
        // ai: Receive again, a file, Receive again) ran its descriptor pool out: "Reading stopped"
        synchronized(lock) { if (handle != 0L) Native.cameraClosed(handle) }
        r.close()
    }

    private fun closeCamera() {
        gen++
        rezoom = null
        opening = false
        if (cameraSlow != null) main.post { cameraSlow = null }
        if (device != null) Log.i(TAG, "close camera")
        try { session?.close() } catch (_: Exception) {}
        session = null
        try { device?.close() } catch (_: Exception) {}
        device = null
        closeReader()
    }

    companion object {
        const val TAG = "lizard"
        // ai: The reader's pool. A camera buffer is held until the GPU has read it (zero copy), and the GPU reads it
        // ai: only after the batch ahead of it: up to the batcher's 400 ms cap, 24 frames at 60 a second. With 16, a
        // ai: throttled GPU (batches of 250 ms) left the camera no buffer and it delivered 45 to 52 frames a second
        // ai: (S26, 2026-09-30).
        const val MAX_IMAGES = 32
    }
}
