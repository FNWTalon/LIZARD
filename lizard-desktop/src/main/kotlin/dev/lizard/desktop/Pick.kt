package dev.lizard.desktop

import kotlin.math.PI
import kotlin.math.sqrt

// ai: Which format a room holds: liblizard/sim/lizard_pick.mjs pickVersion (lizard-android/.../Pick.kt the same), the room the code
// ai: area's device pixels; the largest whole number of blocks whose top ring keeps T_DISPLAY_CYCLE device px a cycle,
// ai: the margin the codec paints (FOCUS_QUIET, 2 modules) counted. A code takes its side and the Margin on both sides of
// ai: the area: two codes two margins apart, the area's edges a margin from them (lizard-web/send.mjs room the same).
object Pick {
    val VERSIONS = (1..128).map { 8 * it }
    // ai: liblizard/src/focus.c FOCUS_RING (32, 64, 128, 256 since 2026-10-10, the 96 before in the 256's place)
    val RINGS = intArrayOf(32, 64, 128, 256)
    const val RING_DEFAULT = 2
    // ai: The Margin (2026-10-23; a gap between two codes before): the light round a code, modules from its ring, the
    // ai: codec's quiet zone (QUIET, FOCUS_QUIET) at least; MARGIN_MAX a painter's widest gap (64 modules), MARGIN_DEFAULT
    // ai: the encoders' own gap (liblizard/gpu/encoder.mjs GAP_MODULES, 12) as a margin
    const val QUIET = 2
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
    fun gapOf(margin: Int) = 2 * (margin - QUIET)
    // ai: the light the presenter keeps at each edge past a frame's quiet zone, a share of the frame's height (a painted
    // ai: symbol's side) in the ring of span cells
    fun edgeShare(span: Int, margin: Int) = (margin - QUIET).toDouble() / (span + 2 * OB_MARGIN + 2 * QUIET)
    // ai: a symbol's room in an area of w x h device px: its painted side where each code takes that and two margins
    fun room(w: Int, h: Int, codes: Int, ring: Int, margin: Int) =
        maxOf(64.0, minOf(w.toDouble() / codes, h.toDouble()) * (modules(ring) + 2.0 * QUIET) / (modules(ring) + 2.0 * margin))
    fun pick(roomPx: Double, ring: Int = RING_DEFAULT): Int {
        var subch = VERSIONS.first()
        for (v in VERSIONS) if (roomFor(v, ring) <= roomPx && v > subch) subch = v
        return subch
    }
}
