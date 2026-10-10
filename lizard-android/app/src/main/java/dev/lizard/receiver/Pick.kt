package dev.lizard.receiver

import kotlin.math.PI
import kotlin.math.sqrt

// ai: Which format a room holds (2026-10-01, the app's sender): liblizard/sim/lizard_pick.mjs pickVersion ported, the room
// ai: the only input: the largest of the 128 formats whose top ring keeps T_DISPLAY_CYCLE device px a cycle in roomPx,
// ai: in the default ring (the 128 since 2026-10-01, span 256; the 64 before), the margin the codec paints
// ai: (FOCUS_QUIET, 2 modules) counted. Keep equal to the web's RING_DEFAULT.
object Pick {
    // ai: every whole number of blocks, 1 to 128 (8 sub-channels each), since 2026-10-01 as the web's VERSIONS; 16 to 1024 by 16 before
    val VERSIONS = (1..128).map { 8 * it }
    // ai: liblizard/src/focus.c FOCUS_RING (32, 64, 128, 256 since 2026-10-10, the 96 before in the 256's place)
    private val RINGS = intArrayOf(32, 64, 128, 256)
    const val RING_DEFAULT = 2
    private const val OB_MARGIN = 15
    private const val QUIET = 2
    private const val T_DISPLAY_CYCLE = 2.7
    private val PICTURES = intArrayOf(256, 384, 512, 768, 1024, 1536)

    private fun rRing(subch: Int) = sqrt(2 * 320.0 * subch / PI)
    fun nFor(subch: Int) = PICTURES.firstOrNull { it >= 3 * rRing(subch) } ?: PICTURES.last()
    fun span(ring: Int = RING_DEFAULT) = 2 * RINGS[ring]
    // ai: a gap of `gap` modules between two codes as a share of one code's side, its margin in (the web's gapFrac)
    fun gapShare(gap: Int, ring: Int = RING_DEFAULT) = gap / (modules(ring) + 2.0 * QUIET)
    private fun modules(ring: Int) = span(ring) + 2 * OB_MARGIN
    fun roomFor(subch: Int, ring: Int = RING_DEFAULT) = (modules(ring) + 2.0 * QUIET) / span(ring) * T_DISPLAY_CYCLE * rRing(subch)
    fun pick(roomPx: Double, ring: Int = RING_DEFAULT): Int {
        var subch = VERSIONS.first()
        for (v in VERSIONS) if (roomFor(v, ring) <= roomPx && v > subch) subch = v
        return subch
    }
}
