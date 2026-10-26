package dev.lizard.pick

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sqrt

// ai: Which format a room holds (liblizard/sim/lizard_pick.mjs pickVersion, liblizard/api/format.cpp liz_pick): the
// ai: largest of the 127 formats whose top ring keeps T_DISPLAY_CYCLE device px a cycle in the room, the margin the
// ai: codec paints (FOCUS_QUIET, 2 modules) counted; where none fits, LIZARD-16. The Android app's and the desktop app's
// ai: senders compile this file (liblizard/kotlin); the picture size is the sender's own (focus_n_for), never set here.
// ai: Pure: no Android in here.
object Pick {
    // ai: every whole number of blocks from two, 2 to 128 (8 sub-channels each): LIZARD-16 to -1024, the web's VERSIONS
    val VERSIONS = (2..128).map { 8 * it }
    // ai: liblizard/src/focus.h FOCUS_RING: 64, 128 and 256 band cells a side; FOCUS_RING_DEFAULT, the 128
    val RINGS = intArrayOf(64, 128, 256)
    const val RING_DEFAULT = 1
    // ai: the margin the codec paints round a symbol, modules (src/focus.h FOCUS_QUIET)
    const val QUIET = 2
    // ai: the pictures a second offered: 60 and its divisors to 10, every picture a whole number of refreshes at 60 Hz
    val FPS = listOf(60, 30, 20, 15, 10)
    // ai: a rate kept from a slider: the nearest offered, a tie to the higher
    fun fpsOf(v: Int) = FPS.minWith(compareBy<Int> { abs(it - v) }.thenByDescending { it })
    // ai: the border's depth past the ring's span, modules a side (src/layout.h OB_MARGIN)
    const val OB_MARGIN = 15
    private const val T_DISPLAY_CYCLE = 2.7

    // ai: the radius subch x 320 coefficients reach, cycles (lizard_pick.mjs R_RING)
    private fun rRing(subch: Int) = sqrt(2 * 320.0 * subch / PI)
    // ai: the ring's span, 2 x its cells
    fun span(ring: Int = RING_DEFAULT) = 2 * RINGS[ring]
    // ai: a symbol's side in modules, its margin not counted
    fun modules(ring: Int = RING_DEFAULT) = span(ring) + 2 * OB_MARGIN
    // ai: the device px a symbol of subch sub-channels needs, its margin in
    fun roomFor(subch: Int, ring: Int = RING_DEFAULT) = (modules(ring) + 2.0 * QUIET) / span(ring) * T_DISPLAY_CYCLE * rRing(subch)
    fun pick(roomPx: Double, ring: Int = RING_DEFAULT): Int {
        var subch = VERSIONS.first()
        for (v in VERSIONS) if (roomFor(v, ring) <= roomPx && v > subch) subch = v
        return subch
    }
}
