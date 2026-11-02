package dev.lizard.receiver

// ai: The app's log tag (logcat -s lizard; the session's native log writes under it too)
internal const val TAG = "lizard"

// ai: libliz.so (src/main/cpp/jni.cpp): the sender (liblizard/core/tx/sender.h) behind a handle, and a system property.
// ai: The receiver is the receive session's (dev.lizard.Session, the binding's: its natives in this same library, which
// ai: registers them at its load), so this object is loaded before any Session is made (load).
object Native {
    init { System.loadLibrary("liz") }

    // ai: the library loaded (and the session's natives registered with it)
    fun load() {}

    // ai: a system property ("" when unset), for switches set over adb while the app runs
    @JvmStatic external fun prop(name: String): String

    // ai: The sender (2026-10-01; liblizard/core/tx/sender.h, jni.cpp's Tx): a file (its path in the app's cache; "" for the test
    // ai: stream) to a handle, 0 with txError() saying why; the format to paint ("" or the codec's refusal; painter 0 the
    // ai: CPU, 1 the GPU, 2 auto, assets the generated tree the GPU's kernels come from); the next frame into the
    // ai: surface's own queue when it is painted (txPresent, on the main thread at the display's pace: below Android 13,
    // ai: or once txPost has refused); the stats JSON; the handle let go.
    @JvmStatic external fun txCreate(path: String, name: String, type: String): Long
    @JvmStatic external fun txError(): String
    @JvmStatic external fun txConfigure(h: Long, subch: Int, span: Int, fps: Int, threads: Int, painter: Int, assets: String, codes: Int, gap: Int, vsynced: Boolean, vsync: Boolean): String
    @JvmStatic external fun txPrepare(h: Long, assets: String): String
    // ai: the blocks a frame of subch sub-channels carries (the format's rate profile, src/lizard_phy.h lizard_blocks_for)
    @JvmStatic external fun blocksFor(subch: Int): Int
    @JvmStatic external fun txPresent(h: Long, surface: android.view.Surface): Boolean
    // ai: the vsync path's post (2026-10-07, jni.cpp txPost): a picture due at the frame timeline `vsync` (expected on
    // ai: screen at `expected` ns, the display's period `vsyncNs`, into the view's w x h2 pixels): the next GPU-filled
    // ai: buffer posted for it, or the CPU-copied present where the device has no GPU ring. 1 posted, 0 none this vsync,
    // ai: -1 the vsync present refused for the rest of this sender (why in txPostStats' error: the window's queue then);
    // ai: and the ring's tally since the last call (JSON)
    @JvmStatic external fun txPost(h: Long, surface: android.view.Surface, vsync: Long, expected: Long, vsyncNs: Long, w: Int, h2: Int): Int
    @JvmStatic external fun txPostStats(h: Long): String
    // ai: the view's surface destroyed (SurfaceHolder.Callback.surfaceDestroyed): its window and rings let go, since the
    // ai: same Surface object comes back holding another window once the view is attached again (2026-10-27)
    @JvmStatic external fun txSurfaceGone(h: Long)
    @JvmStatic external fun txStats(h: Long): String
    @JvmStatic external fun txDestroy(h: Long)
}
