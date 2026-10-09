// ai: The Kotlin binding end to end on the JVM (2026-10-03): the arithmetic, the test stream in
// ai: every ring, one code and two (on white, read by the layouts' regions), a file through two codes into memory (its
// ai: root the sender's, its bytes), a file through one code into a directory, and the exceptions a bad call throws.
package dev.lizard

import java.io.File
import java.nio.ByteBuffer
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class RoundTripTest {
  // ai: the frame on white a quarter taller than it, square for one code, 2:1 for two (the C test's canvas)
  private fun canvas(frame: ByteArray, g: Geometry, codes: Int): Triple<ByteArray, Int, Int> {
    val ch = g.height + g.height / 4
    val cw = codes * ch
    val c = ByteArray(cw * ch) { -1 }
    val x0 = (cw - g.width) / 2
    val y0 = (ch - g.height) / 2
    for (y in 0 until g.height) frame.copyInto(c, (y0 + y) * cw + x0, y * g.width, (y + 1) * g.width)
    return Triple(c, cw, ch)
  }

  @Test fun arithmetic() {
    assertEquals(1, Lizard.abi)
    assertTrue(Lizard.version.matches(Regex("""\d+\.\d+\.\d+""")))
    assertEquals(128, Lizard.ringCells())
    val g = Lizard.geometry(Format(60, codes = 2))
    assertEquals(Geometry(1024, 256, 4, 1160, 2368, 1160, 48, 120), g)
    assertEquals(58, Lizard.pick(1920.0, 1080.0, codes = 2))
    assertEquals(2, Lizard.layoutRects(1920, 1080, 2).size)
  }

  @Test fun testStream() {
    for (codes in 1..2) for (ring in 0 until Lizard.RINGS) {
      // ai: a fixed first id: the C's own misses of a clean frame (about 1 in 200) are measured elsewhere
      FrameSender(Tx.test(1 + 1000 * ring + 100 * codes), Format(26, ring, codes = codes)).use { s ->
        FrameReceiver(layout = codes).use { r ->
          repeat(3) {
            val f = s.frame()
            val (c, cw, ch) = canvas(f.data, s.geometry, codes)
            val p = r.push(c, cw, ch)
            assertEquals(s.geometry.frameBlocks, p.decoded.sumOf { it.count }, "ring $ring, $codes codes")
            assertTrue(p.verdict.test && p.verdict.bad == 0)
            assertEquals(26, r.decoder.held.value)
          }
        }
      }
    }
  }

  @Test fun fileInMemory() {
    val bytes = Random(7).nextBytes(5_000_000)
    FrameSender(Tx.file(bytes, "k.bin", "application/octet-stream"), Format(60, codes = 2)).use { s ->
      FrameReceiver(layout = 2).use { r ->
        var frames = 0
        while (r.file == null && frames++ < 400) {
          val (c, cw, ch) = canvas(s.frame().data, s.geometry, 2)
          r.push(c, cw, ch)
        }
        val file = assertNotNull(r.file, "whole after $frames frames")
        assertEquals(s.info.root, file.root)
        assertEquals("k.bin", file.name)
        assertContentEquals(bytes, file.bytes)
      }
    }
  }

  @Test fun fileInDirectory() {
    val dir = File(System.getProperty("java.io.tmpdir"), "lizard-kotlin-test-${ProcessHandle.current().pid()}")
    val bytes = Random(8).nextBytes(1_500_000)
    Tx.file(bytes, "d.bin").use { tx ->
      Encoder(Format(40)).use { enc ->
        Decoder().use { dec ->
          Rx(dir.path).use { rx ->
            var i = 0
            val buf = ByteBuffer.allocateDirect(enc.geometry.width * enc.geometry.height)
            while (!rx.progress.done && i < 300) {
              enc.paint(tx.next(enc.geometry.frameBlocks), i++, buf, enc.geometry.width, PixelFormat.GREY)
              val d = dec.decode(buf, enc.geometry.width, enc.geometry.height, PixelFormat.GREY, enc.geometry.width)
              rx.frame(d.blocks)
            }
            assertTrue(rx.progress.done, "whole after $i frames")
            assertContentEquals(bytes, File(rx.path).readBytes())
            assertEquals(tx.info.root, rx.root)
          }
        }
      }
    }
    dir.deleteRecursively()
  }

  @Test fun failures() {
    assertFailsWith<IllegalArgumentException> { Encoder(Format(0)) }
    assertFailsWith<IllegalArgumentException> { Lizard.geometry(Format(10, ring = 7)) }
    val e = Encoder(Format(8))
    e.close()
    assertFailsWith<IllegalStateException> { e.paint(ByteArray(Lizard.BLOCK), 0) }
    // ai: arrays too short for what a call writes or reads throw, and nothing is written past their end
    // ai: (2026-10-03: a 16-byte out was written 336,400 bytes and the JVM died later)
    Encoder(Format(8)).use { enc ->
      assertFailsWith<IllegalArgumentException> { enc.paint(ByteArray(8 * Lizard.BLOCK), 0, ByteArray(16)) }
      assertFailsWith<IllegalArgumentException> { enc.paint(ByteArray(10), 0) }
    }
    Tx.test(1).use { tx -> assertFailsWith<IllegalArgumentException> { tx.next(2, ByteArray(10)) } }
    Decoder().use { d -> assertFailsWith<IllegalArgumentException> { d.decode(ByteArray(100), 64, 64) } }
    Rx().use { rx -> assertFailsWith<IllegalArgumentException> { rx.frame(ByteArray(10), 2) } }
  }

  @Test fun engines() {
    val bytes = Random(9).nextBytes(1_000_000)
    val f = Format(40)
    val g = Lizard.geometry(f)
    val store = File(System.getProperty("java.io.tmpdir"), "lizard-kotlin-engine-${ProcessHandle.current().pid()}")
    var released = 0
    Sender(bytes, "e.bin").use { s ->
      s.configure(f, Painter.CPU, threads = 2)
      Receiver(store.path, decoder = DecoderKind.CPU, release = { released++ }).use { r ->
        val frame = ByteArray(g.width * g.height)
        var shown = 0
        val t0 = System.nanoTime()
        while (r.file == null && shown < 600 && System.nanoTime() - t0 < 60_000_000_000L) {
          if (!s.take(frame)) { Thread.sleep(1); continue }
          // ai: on white as a camera sees it, pushed at about 60 a second
          val (c, cw, ch) = canvas(frame, g, 1)
          r.push(c, cw, ch, tag = shown.toLong())
          shown++
          Thread.sleep(16)
        }
        repeat(100) { if (r.file == null) Thread.sleep(20) }
        val path = assertNotNull(r.file, "whole after $shown frames: ${r.stats.take(200)}")
        assertContentEquals(bytes, File(path).readBytes())
        assertTrue(released >= shown - 1, "frames released: $released of $shown")
      }
    }
    store.deleteRecursively()
  }
}

