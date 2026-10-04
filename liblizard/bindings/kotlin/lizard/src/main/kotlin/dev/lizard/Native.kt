// ai: The JNI methods (api/jni.cpp registers them in JNI_OnLoad), and where the library comes from: the system property
// ai: lizard.library (a path: tests, a build tree), else the platform's own (Android's APK, java.library.path), else
// ai: this jar's resources (lizard/native/<os>-<arch>/), extracted once to a folder named by their hash under the user's
// ai: cache (the desktop sender's scheme), then loaded.
package dev.lizard

import java.io.File
import java.nio.ByteBuffer
import java.security.MessageDigest

internal object Native {
  init { load() }

  private fun load() {
    System.getProperty("lizard.library")?.let { System.load(it); return }
    try { System.loadLibrary("lizard"); return } catch (_: UnsatisfiedLinkError) {}
    val os = System.getProperty("os.name").lowercase().let {
      when { it.contains("win") -> "windows"; it.contains("mac") -> "macos"; else -> "linux" }
    }
    val arch = System.getProperty("os.arch").lowercase().let { if (it == "amd64" || it == "x86_64") "x86_64" else if (it == "aarch64" || it == "arm64") "aarch64" else it }
    val name = when (os) { "windows" -> "lizard.dll"; "macos" -> "liblizard.dylib"; else -> "liblizard.so" }
    val res = "/lizard/native/$os-$arch/$name"
    val bytes = Native::class.java.getResourceAsStream(res)?.use { it.readBytes() }
      ?: throw UnsatisfiedLinkError("no $res in the jar, and no lizard on java.library.path")
    val hash = MessageDigest.getInstance("SHA-256").digest(bytes).take(8).joinToString("") { "%02x".format(it) }
    val dir = File(System.getProperty("user.home"), ".cache/lizard/$hash").apply { mkdirs() }
    val lib = File(dir, name)
    if (!lib.exists() || lib.length() != bytes.size.toLong()) {
      val tmp = File.createTempFile(name, ".part", dir)
      tmp.writeBytes(bytes)
      if (!tmp.renameTo(lib)) { lib.delete(); tmp.renameTo(lib) }
    }
    System.load(lib.absolutePath)
  }

  @JvmStatic external fun version(): String
  @JvmStatic external fun abi(): Int
  @JvmStatic external fun simd(): Boolean
  @JvmStatic external fun ringCells(ring: Int): Int
  @JvmStatic external fun geometry(blocks: Int, ring: Int, fps: Int, codes: Int, out: IntArray)
  @JvmStatic external fun roomFor(blocks: Int, ring: Int): Double
  @JvmStatic external fun pick(w: Double, h: Double, codes: Int, ring: Int, top: Int): Int
  @JvmStatic external fun streamFill(id: Int, out: ByteArray)
  @JvmStatic external fun layoutRects(w: Int, h: Int, layout: Int, out: IntArray): Int
  @JvmStatic external fun encoderNew(blocks: Int, ring: Int, fps: Int, codes: Int): Long
  @JvmStatic external fun encoderFree(h: Long)
  @JvmStatic external fun encoderPaint(h: Long, blocks: ByteArray, picture: Int, out: ByteArray, stride: Int, fmt: Int)
  @JvmStatic external fun encoderPaintBuffer(h: Long, blocks: ByteArray, picture: Int, out: ByteBuffer, stride: Int, fmt: Int)
  @JvmStatic external fun decoderNew(nmax: Int): Long
  @JvmStatic external fun decoderFree(h: Long)
  @JvmStatic external fun decoderMaxBlocks(h: Long): Int
  @JvmStatic external fun decode(h: Long, px: ByteArray, offset: Int, w: Int, hh: Int, stride: Int, fmt: Int, held: IntArray, verified: ByteArray, ints: IntArray, floats: FloatArray): Int
  @JvmStatic external fun decodeBuffer(h: Long, px: ByteBuffer, offset: Int, w: Int, hh: Int, stride: Int, fmt: Int, held: IntArray, verified: ByteArray, ints: IntArray, floats: FloatArray): Int
  @JvmStatic external fun txNew(bytes: ByteArray, name: String?, type: String?): Long
  @JvmStatic external fun txNewPath(path: String, name: String?, type: String?): Long
  @JvmStatic external fun txNewTest(first: Int): Long
  @JvmStatic external fun txFree(h: Long)
  @JvmStatic external fun txNext(h: Long, n: Int, out: ByteArray): Int
  @JvmStatic external fun txInfo(h: Long, out: LongArray, root: ByteArray)
  @JvmStatic external fun rxNew(dir: String?, maxBytes: Long): Long
  @JvmStatic external fun rxFree(h: Long)
  @JvmStatic external fun rxFrame(h: Long, blocks: ByteArray, count: Int, out: IntArray)
  @JvmStatic external fun rxProgress(h: Long, ints: IntArray, longs: LongArray, doubles: DoubleArray)
  @JvmStatic external fun rxChunks(h: Long): ByteArray
  @JvmStatic external fun rxMeta(h: Long, which: Int): String
  @JvmStatic external fun rxData(h: Long): ByteArray?
  @JvmStatic external fun rxClear(h: Long)
  @JvmStatic external fun receiverNew(assets: String?, cacheDir: String?, storeDir: String, device: String?, precision: String?, decoder: Int, cpuThreads: Int, layout: Int, release: Receiver.Release, log: Receiver.Log?): Long
  @JvmStatic external fun receiverFree(h: Long)
  @JvmStatic external fun receiverPush(h: Long, luma: ByteArray, offset: Int, w: Int, hh: Int, stride: Int, ts: Long, tag: Long): Int
  @JvmStatic external fun receiverPushBuffer(h: Long, luma: ByteBuffer, offset: Int, w: Int, hh: Int, stride: Int, ts: Long, tag: Long): Int
  @JvmStatic external fun receiverPushHardwareBuffer(h: Long, hb: Any, w: Int, hh: Int, ts: Long, tag: Long): Int
  @JvmStatic external fun receiverWantsLuma(h: Long): Boolean
  @JvmStatic external fun receiverStats(h: Long): String
  @JvmStatic external fun receiverSeries(h: Long, since: Double): DoubleArray
  @JvmStatic external fun receiverSoon(h: Long, on: Boolean)
  @JvmStatic external fun receiverBatchCap(h: Long, n: Int)
  @JvmStatic external fun receiverFile(h: Long): String
  @JvmStatic external fun receiverClear(h: Long)
  @JvmStatic external fun receiverCameraClosed(h: Long)
  @JvmStatic external fun senderNew(bytes: ByteArray?, name: String?, type: String?): Long
  @JvmStatic external fun senderFree(h: Long)
  @JvmStatic external fun senderConfigure(h: Long, blocks: Int, ring: Int, fps: Int, codes: Int, painter: Int, threads: Int, assets: String?, device: String?)
  @JvmStatic external fun senderReady(h: Long): Boolean
  @JvmStatic external fun senderTake(h: Long, out: ByteArray, stride: Int, fmt: Int): Boolean
  @JvmStatic external fun senderTakeBuffer(h: Long, out: ByteBuffer, stride: Int, fmt: Int): Boolean
  @JvmStatic external fun senderStats(h: Long): String
}
