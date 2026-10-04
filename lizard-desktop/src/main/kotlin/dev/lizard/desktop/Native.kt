package dev.lizard.desktop

import java.io.File
import java.security.MessageDigest

// ai: The native sender (lizard-desktop/native/tx_jni.cpp over liblizard/core's lizard_tx): a sender on a mapped file or the
// ai: test stream, its painters (the C on the CPU, or the Vulkan painter: liblizard/core/tx/gpu_painter), and the
// ai: presenter (lizard-desktop/native/presenter.cpp), which takes the sender's frames in order and shows them on the code
// ai: area's canvas through its own Vulkan swapchain, FIFO. load() first: it extracts the bundled library and the GPU
// ai: painter's kernels and tables (build.gradle.kts `bundle`) to a folder named by their hash
// ai: (~/.cache/lizard-sender/<hash>), loads the JDK's jawt (the presenter finds the canvas's window through it, and
// ai: the library links it) and then the library.
object Native {
    // ai: path: the file to send ("" for the test stream); name and type as its header carries them; 0, why in txError()
    @JvmStatic external fun txCreate(path: String, name: String, type: String): Long
    @JvmStatic external fun txError(): String
    @JvmStatic external fun txPrepare(h: Long, assets: String): String
    // ai: painter: 0 the CPU, 1 the GPU, 2 auto; gap: modules between two codes; "" or the refusal
    @JvmStatic external fun txConfigure(h: Long, n: Int, subch: Int, span: Int, fps: Int, codes: Int, gap: Int, threads: Int, painter: Int, assets: String): String
    @JvmStatic external fun txSide(h: Long): Int
    @JvmStatic external fun txWidth(h: Long): Int
    @JvmStatic external fun txReady(h: Long): Boolean
    @JvmStatic external fun txStats(h: Long): String
    @JvmStatic external fun txDestroy(h: Long)

    // ai: The presenter on a displayable canvas, taking the sender tx's frames: 0, why in txError(). size the code's share
    // ai: of the area's room (0.25 to 1), whole whole device pixels a sample (nearest) or stretched (linear), device a
    // ai: substring of a Vulkan device's name ("" the first, LIZ_VK_DEVICE where set).
    @JvmStatic external fun presentStart(canvas: java.awt.Component, tx: Long, fps: Int, size: Float, whole: Boolean, device: String): Long
    @JvmStatic external fun presentFps(p: Long, fps: Int)
    @JvmStatic external fun presentSize(p: Long, size: Float)
    @JvmStatic external fun presentWhole(p: Long, on: Boolean)
    @JvmStatic external fun presentPause(p: Long, on: Boolean)
    // ai: the compositor bypass (X11's _NET_WM_BYPASS_COMPOSITOR on the canvas's top-level window), with full screen
    @JvmStatic external fun presentFullscreen(p: Long, on: Boolean)
    @JvmStatic external fun presentStats(p: Long): String
    @JvmStatic external fun presentStop(p: Long)

    // ai: the GPU painter's tree (setup/send.json, blobs/, spv/send_*.spv), as liblizard/out holds it
    lateinit var assets: String
        private set

    fun load() {
        val names = javaClass.getResourceAsStream("/lizard/files.txt")?.bufferedReader()?.readLines()?.filter { it.isNotBlank() }
            ?: error("the app carries no native files (build.gradle.kts bundle)")
        val bytes = names.associateWith { javaClass.getResourceAsStream("/lizard/$it")!!.readBytes() }
        val md = MessageDigest.getInstance("SHA-256")
        for ((k, v) in bytes) { md.update(k.toByteArray()); md.update(v) }
        val hash = md.digest().joinToString("") { "%02x".format(it) }.take(16)
        val root = File(System.getProperty("user.home"), ".cache/lizard-sender")
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
        val lib = names.first { it.startsWith("native/") }
        System.loadLibrary("jawt")
        System.load(File(dir, lib).absolutePath)
    }
}
