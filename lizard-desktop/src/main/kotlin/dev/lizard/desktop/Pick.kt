package dev.lizard.desktop

import kotlin.math.PI
import kotlin.math.sqrt

// ai: Which format a room holds: liblizard/sim/lizard_pick.mjs pickVersion (lizard-android/.../Pick.kt the same), the room the code
// ai: area's device pixels; the largest whole number of blocks whose top ring keeps T_DISPLAY_CYCLE device px a cycle,
// ai: the margin the codec paints (FOCUS_QUIET, 2 modules) counted. With two codes a symbol's room is the area's width
// ai: over two and the gap's share (lizard-web/send.mjs gapFrac: the gap in modules, GAP_DEFAULT unless set), against the height.
object Pick {
    val VERSIONS = (1..128).map { 8 * it }
    val RINGS = intArrayOf(32, 64, 96, 128)
    const val RING_DEFAULT = 3
    // ai: the gap between two codes, modules (liblizard/gpu/encoder.mjs GAP_MODULES; liblizard/core/tx/send_tables.h the same)
    const val GAP_DEFAULT = 12
    private const val OB_MARGIN = 15
    private const val QUIET = 2
    private const val T_DISPLAY_CYCLE = 2.7
    private val PICTURES = intArrayOf(256, 384, 512, 768, 1024, 1536)

    private fun rRing(subch: Int) = sqrt(2 * 320.0 * subch / PI)
    fun nFor(subch: Int) = PICTURES.firstOrNull { it >= 3 * rRing(subch) } ?: PICTURES.last()
    fun span(ring: Int = RING_DEFAULT) = 2 * RINGS[ring]
    private fun modules(ring: Int) = span(ring) + 2 * OB_MARGIN
    fun roomFor(subch: Int, ring: Int = RING_DEFAULT) = (modules(ring) + 2.0 * QUIET) / span(ring) * T_DISPLAY_CYCLE * rRing(subch)
    fun gapFrac(codes: Int, ring: Int, gap: Int) = if (codes > 1) (codes - 1).toDouble() * gap / (modules(ring) + 2 * QUIET) else 0.0
    // ai: a symbol's room in an area of w x h device px
    fun room(w: Int, h: Int, codes: Int, ring: Int, gap: Int) = maxOf(64.0, minOf(w / (codes + gapFrac(codes, ring, gap)), h.toDouble()))
    fun pick(roomPx: Double, ring: Int = RING_DEFAULT): Int {
        var subch = VERSIONS.first()
        for (v in VERSIONS) if (roomFor(v, ring) <= roomPx && v > subch) subch = v
        return subch
    }
}
