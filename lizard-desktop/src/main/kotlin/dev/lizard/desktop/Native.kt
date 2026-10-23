package dev.lizard.desktop

import java.io.File
import java.security.MessageDigest

// ai: The native sender (lizard-desktop/native/tx_jni.cpp over liblizard/core's lizard_tx): a sender on a mapped file or the
// ai: test stream, its painters (the C on the CPU, or the Vulkan painter: liblizard/core/tx/gpu_painter), and the
// ai: presenter (lizard-desktop/native/presenter.cpp), which takes the sender's frames in order and shows them on the code
// ai: area's canvas through its own Vulkan swapchain, FIFO. load() first: it extracts the bundled library and the GPU
// ai: painter's kernels and tables (build.gradle.kts `bundle`) to a folder named by their hash under the cache
// ai: (Native.cache: ~/.cache/lizard-sender on Linux), loads the JDK's jawt (the presenter finds the canvas's window
// ai: through it) and then the library. On macOS the bundle also carries MoltenVK, put beside the library, where the
// ai: library's Vulkan loader finds it.
object Native {
    // ai: path: the file to send ("" for the test stream); name and type as its header carries them; 0, why in txError()
    @JvmStatic external fun txCreate(path: String, name: String, type: String): Long
    @JvmStatic external fun txError(): String
    @JvmStatic external fun txPrepare(h: Long, assets: String): String
    // ai: painter: 0 the CPU, 1 the GPU, 2 auto; gap: modules between two codes; "" or the refusal
    @JvmStatic external fun txConfigure(h: Long, n: Int, subch: Int, span: Int, fps: Int, codes: Int, gap: Int, threads: Int, painter: Int, assets: String): String
    // ai: the blocks a frame of subch sub-channels carries (the format's rate profile, src/focus.h focus_blocks_for)
    @JvmStatic external fun blocksFor(subch: Int): Int
    @JvmStatic external fun txStats(h: Long): String
    @JvmStatic external fun txDestroy(h: Long)

    // ai: The presenter on a displayable canvas, taking the sender tx's frames: 0, why in txError(). size the code's share
    // ai: of the area's room (0.25 to 1), edge the light kept at each edge past the frame's own (Pick.edgeShare), whole
    // ai: whole device pixels a sample (nearest) or stretched (linear), device a substring of a Vulkan device's name ("" the
    // ai: first, LIZ_VK_DEVICE where set).
    // ai: place: macOS's layer over the canvas (CodeCanvas.place), ignored elsewhere; presentPlace when the canvas moves
    @JvmStatic external fun presentStart(canvas: java.awt.Component, tx: Long, fps: Int, size: Float, edge: Float, whole: Boolean, place: FloatArray): Long
    @JvmStatic external fun presentPlace(p: Long, place: FloatArray)
    // ai: the desktop's file chooser (2026-10-23, native/portal.cpp): the window manager's window around a canvas (0 none,
    // ai: on the UI thread), and the XDG portal's OpenFile attached to it (Windows: the shell's dialog), blocking until it
    // ai: closes (never on the UI thread): the file:// URIs chosen (Windows: paths), none when cancelled, null where it
    // ai: could not be asked (why: txError)
    @JvmStatic external fun windowOf(canvas: java.awt.Component): Long
    @JvmStatic external fun portalPick(window: Long, title: String, directory: Boolean, multiple: Boolean): Array<String>?
    // ai: the window manager's own full screen for the window windowOf gave (2026-10-23): asked (false where it could not
    // ai: be: not X11), and read
    @JvmStatic external fun windowFullscreen(window: Long, on: Boolean): Boolean
    @JvmStatic external fun windowIsFullscreen(window: Long): Boolean
    @JvmStatic external fun presentFps(p: Long, fps: Int)
    @JvmStatic external fun presentSize(p: Long, size: Float, edge: Float)
    @JvmStatic external fun presentPause(p: Long, on: Boolean)
    @JvmStatic external fun presentStats(p: Long): String
    @JvmStatic external fun presentStop(p: Long)

    // ai: the GPU painter's tree (setup/send.json, blobs/, spv/send_*.spv), as liblizard/out holds it
    lateinit var assets: String
        private set

    private val os = System.getProperty("os.name").lowercase()
    val mac = os.contains("mac")
    val windows = os.contains("win")
    // ai: the app's cache, where each platform keeps one: the native files' folders and the bundles' tars (Windows: not
    // ai: in %LOCALAPPDATA%\LIZARD, the MSI's folder)
    val cache: File = when {
        windows -> File(System.getenv("LOCALAPPDATA")?.takeIf { it.isNotEmpty() } ?: System.getProperty("user.home"), "lizard-sender")
        mac -> File(System.getProperty("user.home"), "Library/Caches/lizard-sender")
        else -> File(System.getenv("XDG_CACHE_HOME")?.takeIf { it.isNotEmpty() } ?: File(System.getProperty("user.home"), ".cache").path, "lizard-sender")
    }

    fun load() {
        val names = javaClass.getResourceAsStream("/lizard/files.txt")?.bufferedReader()?.readLines()?.filter { it.isNotBlank() }
            ?: error("the app carries no native files (build.gradle.kts bundle)")
        val bytes = names.associateWith { javaClass.getResourceAsStream("/lizard/$it")!!.readBytes() }
        val md = MessageDigest.getInstance("SHA-256")
        for ((k, v) in bytes) { md.update(k.toByteArray()); md.update(v) }
        val hash = md.digest().joinToString("") { "%02x".format(it) }.take(16)
        val root = cache
        val dir = File(root, hash)
        val done = File(dir, ".done")
        if (!done.exists()) {
            for ((k, v) in bytes) File(dir, k).apply { parentFile.mkdirs(); writeBytes(v) }
            done.writeText(hash)
        }
        // ai: the folders of older builds go, but the newest two besides this one's: an instance of an older build still
        // ai: open reads its assets again at each Start (2026-10-04: deleting every other folder took them
        // ai: from under it), and a dev box makes a build a day, about 15 MB each
        root.listFiles()?.filter { it.isDirectory && it.name != hash }?.sortedByDescending { it.lastModified() }?.drop(2)
            ?.forEach { runCatching { it.deleteRecursively() } }
        assets = File(dir, "assets").path
        val lib = names.first { it.startsWith("native/") && it.contains("lizard_desktop") }
        System.loadLibrary("jawt")
        System.load(File(dir, lib).absolutePath)
    }
}
