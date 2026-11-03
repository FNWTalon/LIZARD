package dev.lizard.receiver

import java.io.File

// ai: The GPU's clock ceiling against its top, which the red warning states (2026-10-03): Adreno's kgsl,
// ai: max_gpuclk (Hz) against max_clock_mhz, which the kernel lowers as the phone heats. On the S26 2026-10-03 in 2:1,
// ai: at thermal status 1, the ceiling was 646 of 1,300 MHz and the decoder kept up with 76 to 85 of about 105
// ai: half-frames a second (the rate 2.5 to 1.7 MB/s), where Android's thermal status alone (the warning from moderate,
// ai: 2) said nothing. Readable by the app on the S26; a phone whose files are not gives null, and the warning falls
// ai: back to the thermal status. (The CPU clusters' ceilings fall sooner and recover later, 2.9 of 4.7 GHz at thermal
// ai: status 0 with the GPU at its top, and the decoder is the GPU's: not read.)
object Clocks {
    // ai: allowed and top, MHz
    data class Read(val mhz: Int, val top: Int) {
        // ai: under its top by more than 2%: lowered, not a top a hair under the table's
        val throttled get() = mhz < 0.98 * top
    }

    private fun num(f: File) = runCatching { f.readText().trim().toLong() }.getOrNull()

    fun read(): Read? {
        val g = File("/sys/class/kgsl/kgsl-3d0")
        val m = num(File(g, "max_gpuclk")) ?: return null
        val t = num(File(g, "max_clock_mhz")) ?: return null
        return if (t > 0) Read((m / 1_000_000).toInt(), t.toInt()) else null
    }
}

// ai: The heat warning (2026-10-01): a red triangle and the line "Thermal throttling." Since
// ai: 2026-10-03 it shows when the phone has lowered the GPU's clock ceiling (Clocks.kt, under its top; the CPU's not),
// ai: the clock itself only in the statistics' heat line (2026-11-03: the GPU named nowhere else); where the GPU's
// ai: clock cannot be read, from Android's moderate thermal status (2) up. Since 2026-10-07, first of all, when the
// ai: camera's rate has fallen from its peak (ReceiveState.cameraSlow): "Thermal throttling. Camera at 53 of 60 frames a
// ai: second.", the loss that costs the rate most. heatText is its line, null while nothing throttles.
fun heatText(status: Int, clocks: Clocks.Read?, camera: ReceiveState.CameraSlow? = null): String? = when {
    camera != null -> "Thermal throttling. Camera at ${camera.now} of ${camera.max} frames a second."
    clocks != null -> if (clocks.throttled) "Thermal throttling." else null
    status >= android.os.PowerManager.THERMAL_STATUS_MODERATE -> "Thermal throttling."
    else -> null
}
