// ai: The engines (lizard.h layer 4): a receiver decoding camera frames on threads of its own (the GPU decoder in
// ai: batches, or the C on a pool) into a transfer, and a sender painting frames ahead (the GPU's encoder, or the C on
// ai: several threads). What the Android app and the desktop sender run. Native builds only (a build without them
// ai: throws UnsupportedOperationException).
package dev.lizard

import java.nio.ByteBuffer

enum class DecoderKind(internal val code: Int) { AUTO(0), GPU(1), CPU(2) }
enum class Painter(internal val code: Int) { CPU(0), GPU(1), AUTO(2) }

// ai: storeDir: where the file is kept (its lizard-xfer folder emptied first); assets: the GPU's kernels and nets (null:
// ai: the CPU decoder); device: a substring of the Vulkan device's name; release: a frame pushed is the caller's again
// ai: (called from the receiver's threads, or inside the push); log: its lines (from its threads)
class Receiver(
  storeDir: String,
  assets: String? = null,
  decoder: DecoderKind = DecoderKind.AUTO,
  layout: Int = 1,
  device: String? = null,
  precision: String? = null,
  cacheDir: String? = null,
  cpuThreads: Int = 0,
  log: Log? = null,
  release: Release = Release {},
) : Handle(Native.receiverNew(assets, cacheDir, storeDir, device, precision, decoder.code, cpuThreads, layout, release, log), Native::receiverFree) {
  fun interface Release { fun released(tag: Long) }
  fun interface Log { fun log(line: String) }

  // ai: a luma plane, copied (or dropped) before the call returns
  fun push(luma: ByteArray, w: Int, h: Int, stride: Int = w, timestampNs: Long = System.nanoTime(), tag: Long = 0, offset: Int = 0) {
    Native.receiverPush(live(), luma, offset, w, h, stride, timestampNs, tag)
  }
  fun push(luma: ByteBuffer, w: Int, h: Int, stride: Int, timestampNs: Long, tag: Long, offset: Int = 0) {
    require(luma.isDirect) { "a direct buffer" }
    Native.receiverPushBuffer(live(), luma, offset, w, h, stride, timestampNs, tag)
  }
  // ai: Android: a camera frame's android.hardware.HardwareBuffer, read by the GPU in place until release(tag)
  fun pushHardwareBuffer(buffer: Any, w: Int, h: Int, timestampNs: Long, tag: Long) {
    Native.receiverPushHardwareBuffer(live(), buffer, w, h, timestampNs, tag)
  }
  val wantsLuma: Boolean get() = Native.receiverWantsLuma(live())
  val stats: String get() = Native.receiverStats(live())
  // ai: [the held word's version, then 8 doubles a frame decoded since sinceMs]: lizard.h liz_receiver_series
  fun series(sinceMs: Double): DoubleArray = Native.receiverSeries(live(), sinceMs)
  fun soon(on: Boolean) = Native.receiverSoon(live(), on)
  fun batchCap(frames: Int) = Native.receiverBatchCap(live(), frames)
  // ai: the received file's path once whole and verified, else null
  val file: String? get() = Native.receiverFile(live()).ifEmpty { null }
  fun clear() = Native.receiverClear(live())
  fun cameraClosed() = Native.receiverCameraClosed(live())
}

// ai: a file's bytes (copied in), or the test stream (bytes null)
class Sender(bytes: ByteArray? = null, name: String = "", type: String = "") : Handle(Native.senderNew(bytes, name, type), Native::senderFree) {
  var geometry: Geometry? = null
    private set
  // ai: the format to paint (a re-pick keeps the transfer), on painter (auto: the GPU where its first frames are the
  // ai: C's), threads CPU painters (0: three), the GPU's assets and device
  fun configure(format: Format, painter: Painter = Painter.AUTO, threads: Int = 0, assets: String? = null, device: String? = null) {
    Native.senderConfigure(live(), format.blocks, format.ring, format.fps, format.codes, painter.code, threads, assets, device)
    geometry = Lizard.geometry(format)
  }
  val ready: Boolean get() = Native.senderReady(live())
  // ai: the next frame (width x height of fmt) if painted: true, it is the screen's now; false, not yet
  fun take(out: ByteArray, fmt: PixelFormat = PixelFormat.GREY): Boolean {
    val g = geometry ?: throw IllegalStateException("not configured")
    return Native.senderTake(live(), out, g.width * fmt.bytes, fmt.code)
  }
  fun take(out: ByteBuffer, stride: Int, fmt: PixelFormat): Boolean {
    require(out.isDirect) { "a direct buffer" }
    return Native.senderTakeBuffer(live(), out, stride, fmt.code)
  }
  val stats: String get() = Native.senderStats(live())
}
