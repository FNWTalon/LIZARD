package dev.lizard.desktop

import kotlin.math.PI
import kotlin.math.sqrt

// ai: Which format a room holds: liblizard/sim/lizard_pick.mjs pickVersion (lizard-android/.../Pick.kt the same), the room the code
// ai: area's device pixels; the largest whole number of blocks whose top ring keeps T_DISPLAY_CYCLE device px a cycle,
// ai: the margin the codec paints (FOCUS_QUIET, 2 modules) counted. Two codes stand Spacing apart and the area's edges
// ai: Margin from them, both from the rings (lizard-web/send.mjs room the same).
object Pick {
    val VERSIONS = (1..128).map { 8 * it }
    // ai: liblizard/src/focus.c FOCUS_RING (32, 64, 128, 256 since 2026-10-10, the 96 before in the 256's place)
    val RINGS = intArrayOf(32, 64, 128, 256)
    const val RING_DEFAULT = 2
    // ai: Spacing and Margin (2026-10-23; one Margin, two of them between codes, that morning; a gap between two codes
    // ai: before): the light between two codes' rings, two quiet zones (QUIET, FOCUS_QUIET) at least, and between a ring and
    // ai: the area's edge, one at least, in modules. SPACING_MAX a painter's widest gap (64 modules) and its quiet zones;
    // ai: SPACING_DEFAULT the encoders' own gap (liblizard/gpu/encoder.mjs GAP_MODULES, 12) and its quiet zones
    const val QUIET = 2
    const val SPACING_MIN = 2 * QUIET
    const val SPACING_MAX = 68
    const val SPACING_DEFAULT = 16
    const val MARGIN_MAX = 34
    const val MARGIN_DEFAULT = 8
    private const val OB_MARGIN = 15
    private const val T_DISPLAY_CYCLE = 2.7
    private val PICTURES = intArrayOf(256, 384, 512, 768, 1024, 1536)

    private fun rRing(subch: Int) = sqrt(2 * 320.0 * subch / PI)
    fun nFor(subch: Int) = PICTURES.firstOrNull { it >= 3 * rRing(subch) } ?: PICTURES.last()
    fun span(ring: Int = RING_DEFAULT) = 2 * RINGS[ring]
    private fun modules(ring: Int) = span(ring) + 2 * OB_MARGIN
    fun roomFor(subch: Int, ring: Int = RING_DEFAULT) = (modules(ring) + 2.0 * QUIET) / span(ring) * T_DISPLAY_CYCLE * rRing(subch)
    // ai: the gap a painter puts between two codes past their quiet zones, modules
    fun gapOf(spacing: Int) = spacing - 2 * QUIET
    // ai: the light the presenter keeps at each edge past a frame's quiet zone, a share of the frame's height (a painted
    // ai: symbol's side) in the ring of span cells
    fun edgeShare(span: Int, margin: Int) = (margin - QUIET).toDouble() / (span + 2 * OB_MARGIN + 2 * QUIET)
    // ai: a symbol's room in an area of w x h device px: its painted side, the codes, the spacing between them and the
    // ai: margins at the edges fitted across, the code and the margins down
    fun room(w: Int, h: Int, codes: Int, ring: Int, spacing: Int, margin: Int): Double {
        val across = codes * modules(ring) + (codes - 1) * spacing + 2.0 * margin
        val down = modules(ring) + 2.0 * margin
        return maxOf(64.0, minOf(w / across, h / down) * (modules(ring) + 2.0 * QUIET))
    }
    fun pick(roomPx: Double, ring: Int = RING_DEFAULT): Int {
        var subch = VERSIONS.first()
        for (v in VERSIONS) if (roomFor(v, ring) <= roomPx && v > subch) subch = v
        return subch
    }
}
