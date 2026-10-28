package dev.lizard.receiver

import android.hardware.HardwareBuffer
import android.media.Image
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

// ai: libliz.so (src/main/cpp/jni.cpp), a Receiver (liblizard/core/rx/receiver.h) behind a handle. A pushed frame stays in held
// ai: until the receiver's release(tag) closes it: the GPU reads the camera's buffer in place, so the Image (and the
// ai: HardwareBuffer that references its AHardwareBuffer) outlives push.
object Native {
    init { System.loadLibrary("liz") }

    // ai: decoder auto | gpu | cpu, layout 1:1 | 2:1 (Settings); the precision is the receiver's own choice, auto
    @JvmStatic external fun create(assets: String, cacheDir: String, storeDir: String, decoder: String, layout: String): Long
    // ai: hb (zero copy) or luma (a direct buffer of the Y plane); throws IllegalArgumentException only before the
    // ai: receiver has the frame, and the caller then closes it itself. lens: the camera's report of its lens in the
    // ai: frame's px, [k1, k2, k3, f], or null (Engine.lensOf)
    @JvmStatic external fun push(h: Long, hb: HardwareBuffer?, luma: ByteBuffer?, width: Int, height: Int, stride: Int,
                                 timestampNs: Long, tag: Long, lens: FloatArray?)
    // ai: A frame in a format developed to luma (Engine.develop: P010, RAW16, RAW10, RAW12; jni.cpp): plane 0 (its row and
    // ai: pixel strides), the format, the image's size, the crop of it read (x0, y0, w, h; the whole frame for P010) and
    // ai: bin (1, or 2 for a 2 x 2 bin of a RAW crop), the CFA and levels (Engine.levelsOf), then as push. The receiver
    // ai: develops it to luma (liblizard/core/ingest/develop.h: the GPU decoder on its device, the C on this thread)
    // ai: before push returns, and releases tag as for any frame.
    @JvmStatic external fun pushRaw(h: Long, plane: ByteBuffer, rowStride: Int, pixelStride: Int, format: Int, width: Int, height: Int,
                                    x0: Int, y0: Int, w: Int, hh: Int, bin: Int, cfa: Int, levels: FloatArray, timestampNs: Long, tag: Long, lens: FloatArray?)
    // ai: whether this receiver reads luma bytes (the C on the CPU) rather than the camera's buffer in place
    @JvmStatic external fun wantsLuma(h: Long): Boolean
    @JvmStatic external fun stats(h: Long): String
    // ai: the receiver's newest frames with no JSON (receiver.h series): [the version the last read word names or 0,
    // ai: then 8 doubles a frame captured after sinceMs, oldest first: ms on the camera's clock, verified blocks, new
    // ai: blocks, found (1 or 0), the pilots' r, its standard error, r2, its standard error (NaN where none was read)]
    @JvmStatic external fun series(h: Long, sinceMs: Double): DoubleArray
    // ai: the most frames a GPU launch takes, 1 to 32 (receiver.h batchCap; Settings.batch); a launch goes as soon as a
    // ai: frame is staged and a lane is free whatever this is, never waiting to fill a batch
    @JvmStatic external fun batchCap(h: Long, n: Int)
    @JvmStatic external fun file(h: Long): String
    // ai: the camera's reader closed, every frame of it handed back: the GPU decoder drops its imports of its buffers, and
    // ai: on both decoders the stats' window ends with the stream (its figures 0 until the next stream's first second has
    // ai: closed, that window the next stream's alone: liblizard/include/lizard.h liz_receiver_camera_closed)
    @JvmStatic external fun cameraClosed(h: Long)
    // ai: the transfer in hand forgotten, the held word with it (2026-10-07): the received file deleted from the app
    @JvmStatic external fun clear(h: Long)
    // ai: one transfer forgotten by its root, the one in hand or one parked unfinished (2026-10-22): Home's Delete
    @JvmStatic external fun forget(h: Long, root: String)
    @JvmStatic external fun destroy(h: Long)
    // ai: Save replays (2026-10-03; liblizard/core/rx/replay.h): a replay keeping the newest `frames` frames in dir (made there),
    // ai: handed to a receiver by record (0 takes it back); replayEnd ends its run once the frames the decoder holds for
    // ai: it have come (rxStats the receiver's stats JSON, rows stats.jsonl's lines, more meta.json's own fields) and
    // ai: lets the handle go: {"frames", "bytes", "error"} as JSON (error: a write that ended the frames early), or an
    // ai: IllegalStateException saying why
    @JvmStatic external fun replayNew(dir: String, run: String, frames: Int): Long
    @JvmStatic external fun record(h: Long, replay: Long)
    @JvmStatic external fun replayEnd(replay: Long, rxStats: String, rows: String, more: String): String
    // ai: a system property ("" when unset), for switches set over adb while the app runs
    @JvmStatic external fun prop(name: String): String
    // ai: the PC stream over USB (PcLink): a USB accessory's fd, detached, to the core's bridge, which closes it; false
    // ai: where one is open already
    @JvmStatic external fun streamAccessory(fd: Int): Boolean
    @JvmStatic external fun accessoryOpen(): Boolean

    // ai: The sender (2026-10-01; liblizard/core/tx/sender.h, jni.cpp's Tx): a file (its path in the app's cache; "" for the test
    // ai: stream) to a handle, 0 with txError() saying why; the format to paint ("" or the codec's refusal; painter 0 the
    // ai: CPU, 1 the GPU, 2 auto, assets the generated tree the GPU's kernels come from); the next frame into the
    // ai: surface's own queue when it is painted (txPresent, on the main thread at the display's pace: below Android 13,
    // ai: or once txPost has refused); the stats JSON; the handle let go.
    @JvmStatic external fun txCreate(path: String, name: String, type: String): Long
    @JvmStatic external fun txError(): String
    @JvmStatic external fun txConfigure(h: Long, subch: Int, span: Int, fps: Int, threads: Int, painter: Int, assets: String, codes: Int, gap: Int, vsynced: Boolean, vsync: Boolean): String
    @JvmStatic external fun txPrepare(h: Long, assets: String): String
    // ai: the blocks a frame of subch sub-channels carries (the format's rate profile, src/focus.h focus_blocks_for)
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

    // ai: A frame pushed and not yet released, and the Engine that pushed it. The tags are the process's and the frames
    // ai: each Engine's, so an Engine going away (the activity's onDestroy, which can run after the next activity's
    // ai: Engine has started) touches only its own (2026-10-01: each Engine counted tags from 1 into this one
    // ai: map, and closeAll() closed every Engine's frames, the new one's under its receiver's GPU included).
    class Held(val image: Image, val hb: HardwareBuffer?, val owner: Any) : AutoCloseable {
        override fun close() { hb?.close(); image.close() }
    }
    val held = ConcurrentHashMap<Long, Held>()
    private val tags = AtomicLong(1)
    fun hold(image: Image, hb: HardwareBuffer?, owner: Any): Long = tags.getAndIncrement().also { held[it] = Held(image, hb, owner) }
    fun heldBy(owner: Any) = held.values.count { it.owner === owner }

    // ai: called from a receiver thread (attached by jni.cpp), or from push's own thread when it drops a frame
    @JvmStatic fun release(tag: Long) {
        try { held.remove(tag)?.close() } catch (_: IllegalStateException) {}   // ai: its reader already closed
    }

    fun closeAll(owner: Any) { for ((k, v) in held) if (v.owner === owner) release(k) }
}
