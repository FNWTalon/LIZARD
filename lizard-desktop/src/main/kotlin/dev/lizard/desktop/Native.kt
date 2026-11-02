package dev.lizard.desktop

import java.io.File

// ai: The app's native side (lizard-desktop/native over liblizard/core): the sender (tx_jni.cpp over lizard_tx: a sender on
// ai: a mapped file or the test stream, its painters, the C on the CPU or the Vulkan painter, liblizard/core/tx/
// ai: gpu_painter), the presenter (presenter.cpp, which takes the sender's frames in order and shows them on the code
// ai: area's canvas through its own Vulkan swapchain, FIFO), the desktop's file chooser and full screen; and the
// ai: receive session's natives (liblizard/api/jni_session.cpp, dev.lizard.Session's: a camera into liblizard/core's
// ai: Receiver, the GPU decoder or the C), registered as the library loads (jni_util.cpp JNI_OnLoad). load()
// ai: first: it extracts the bundled library, the GPU painter's kernels and tables and the decoder's setups, blobs and
// ai: modules (build.gradle.kts `bundle`) to a folder under the cache named by the hash the build wrote at the top of
// ai: files.txt (Native.cache: ~/.cache/lizard-sender on Linux), loads the JDK's jawt (the presenter finds the canvas's
// ai: window through it) and then the library. On macOS the bundle also carries MoltenVK, put beside the library, where
// ai: the library's Vulkan loader finds it.
object Native {
    // ai: the process's priority raised as far as a user's may (native/priority.cpp): what was done, "" nothing
    @JvmStatic external fun raisePriority(): String
    // ai: path: the file to send ("" for the test stream); name and type as its header carries them; 0, why in txError()
    @JvmStatic external fun txCreate(path: String, name: String, type: String): Long
    @JvmStatic external fun txError(): String
    @JvmStatic external fun txPrepare(h: Long, assets: String): String
    // ai: painter: 0 the CPU, 1 the GPU, 2 auto; gap: modules between two codes; "" or the refusal
    @JvmStatic external fun txConfigure(h: Long, subch: Int, span: Int, fps: Int, codes: Int, gap: Int, threads: Int, painter: Int, assets: String): String
    // ai: the blocks a frame of subch sub-channels carries (the format's rate profile, src/lizard_phy.h lizard_blocks_for)
    @JvmStatic external fun blocksFor(subch: Int): Int
    @JvmStatic external fun txStats(h: Long): String
    @JvmStatic external fun txDestroy(h: Long)

    // ai: The presenter on a displayable canvas, taking the sender tx's frames on the sender's device (txPrepare): 0,
    // ai: why in txError(). size the code's share of the area's room (0.25 to 1), edge the light kept at each edge past
    // ai: the frame's own (Room.edgeShare); the frame stretched (linear).
    // ai: place: macOS's layer over the canvas (CodeCanvas.place), ignored elsewhere; presentPlace when the canvas moves.
    // ai: vsync: the monitor held at a refresh fps divides while presenting (presenter.h setVsync), presentVsync after.
    @JvmStatic external fun presentStart(canvas: java.awt.Component, tx: Long, fps: Int, size: Float, edge: Float, vsync: Boolean, direct: Boolean, place: FloatArray): Long
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
    // ai: a heavyweight component's native window given its own background, 0xRRGGBB: what X fills it with where it is
    // ai: shown or uncovered before it draws (2026-11-02; AWT's is the desktop theme's grey); false where it could not be
    @JvmStatic external fun windowBackground(component: java.awt.Component, rgb: Int): Boolean
    @JvmStatic external fun presentFps(p: Long, fps: Int)
    @JvmStatic external fun presentVsync(p: Long, on: Boolean)
    @JvmStatic external fun presentSize(p: Long, size: Float, edge: Float)
    @JvmStatic external fun presentPause(p: Long, on: Boolean)
    @JvmStatic external fun presentStats(p: Long): String
    // ai: the picture on screen as grey bytes (presenter.h snapshot), its width and height into wh; null where none
    @JvmStatic external fun presentSnapshot(p: Long, wh: IntArray): ByteArray?
    @JvmStatic external fun presentStop(p: Long)

    // ai: the GPU tree (setup/send.json, the decoder's six setups, blobs/, spv/), as liblizard/out holds it
    lateinit var assets: String
        private set
    // ai: this build's folder under the cache (the pipeline cache's place: old builds' go with their folders)
    lateinit var buildDir: File
        private set
    // ai: the native library's folder there, and the files carried beside it (build.gradle.kts bundle: libusb, the
    // ai: Windows driver helper lizard_usb.exe, Aravis and its libraries)
    val nativeDir get() = File(buildDir, "native")

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
    // ai: The app's data (2026-10-25), never under the cache, whose folders load() prunes: the receiver's store (the
    // ai: transfers kept to resume) and the received files' list. LIZ_DATA names another (the tests').
    val data: File = System.getenv("LIZ_DATA")?.takeIf { it.isNotEmpty() }?.let { File(it) } ?: when {
        windows -> File(System.getenv("LOCALAPPDATA")?.takeIf { it.isNotEmpty() } ?: System.getProperty("user.home"), "LIZARD-data")
        mac -> File(System.getProperty("user.home"), "Library/Application Support/LIZARD")
        else -> File(System.getenv("XDG_DATA_HOME")?.takeIf { it.isNotEmpty() } ?: File(System.getProperty("user.home"), ".local/share").path, "lizard")
    }

    // ai: files.txt (build.gradle.kts bundle): "hash <h>", then the bundle's names. <h>, made at the build since
    // ai: 2026-10-25, names this build's folder under the cache (a launch read and hashed the whole bundle to name it
    // ai: before: 70 MB with a debug library, 22 MB with a release one): once its .done is there no file of the bundle
    // ai: is read; else each is copied out of the jar in turn, .done last.
    fun load() {
        val lines = javaClass.getResourceAsStream("/lizard/files.txt")?.bufferedReader()?.readLines()?.filter { it.isNotBlank() }
            ?: error("the app carries no native files (build.gradle.kts bundle)")
        val hash = lines.firstOrNull()?.let { Regex("hash ([0-9a-f]{16})").matchEntire(it)?.groupValues?.get(1) }
            ?: error("the app's files.txt names no hash (build.gradle.kts bundle)")
        val names = lines.drop(1)
        val root = cache
        val dir = File(root, hash)
        val done = File(dir, ".done")
        if (!done.exists()) {
            for (k in names) {
                val src = javaClass.getResourceAsStream("/lizard/$k") ?: error("the app carries no $k, which its files.txt names")
                src.use { s -> File(dir, k).apply { parentFile.mkdirs() }.outputStream().use { s.copyTo(it, 1 shl 20) } }
            }
            done.writeText(hash)
        }
        // ai: the folders of older builds go, but the newest two besides this one's: an instance of an older build still
        // ai: open reads its assets again at each Start (2026-10-04: deleting every other folder took them
        // ai: from under it), and a dev box makes a build a day, about 15 MB each
        root.listFiles()?.filter { it.isDirectory && it.name != hash }?.sortedByDescending { it.lastModified() }?.drop(2)
            ?.forEach { runCatching { it.deleteRecursively() } }
        assets = File(dir, "assets").path
        buildDir = dir
        val lib = names.first { it.startsWith("native/") && it.contains("lizard_desktop") }
        System.loadLibrary("jawt")
        System.load(File(dir, lib).absolutePath)
    }
}
