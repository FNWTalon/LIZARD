// ai: The app's PhaseLock (app/src/main/java/dev/lizard/receiver/PhaseLock.kt, the class itself, compiled beside
// ai: this file) against a model of what it steers and reads, and against the phone's own recorded frames.
// ai: The model: a camera whose frame is `tcNs` long and takes a delay seven captures after it is asked, as the S26
// ai: does (delays asked while one is on its way queue, one a frame, as Camera2's requests do); a display refreshing every `tdNs`, a
// ai: painted frame every 60 / `painted` refreshes; the receiver's series asked twice a second, 0.7 s behind.
// ai: What a capture reads, two ways:
// ai:   the box (the first model, kept): 60 of 64 blocks unless a refresh that changes the painted frame falls
// ai:     within `winNs` of its start (its exposure and readout), and then 3;
// ai:   measured (`measured`): as the S26 read this desk's monitor on 2026-09-30, by folding a minute of free-running
// ai:     frames by the display's period (research/STATUS.md "The phase lock on the S26, and track"). A capture with
// ai:     no change in its window reads 41 blocks at the edges of the phases that read and 50 at their middle,
// ai:     scattered by 4, and 54 where the picture has been up for more than a refresh; one refresh in four the
// ai:     sender's picture does not change (its paint missed the refresh), so a quarter of the captures off those
// ai:     phases read too; a change within 0.6 ms of either end of the window is read through in proportion; 25
// ai:     captures in 1,000 are short anywhere; one capture in six is not in the series (the receiver did not
// ai:     process it); and a delay lands as whole lines of the sensor (5.1775 us), as the phone's did.
// ai: `rough`: what a hand adds to either: in the box a capture within a millisecond of holding a change is short
// ai: half the time and 3 in 100 of the rest are short anyway; in both, every 23 s the frames are short for 0.6 s.
// ai: `pilots` (research/SPEC.md 7.3; signed since 2026-10-01): the sender flashes, each painted picture's count
// ai: (mod 4) the count of the sender's changes, and each capture reads r (bit 0 of the count: the even blocks) and
// ai: r2 (bit 1: the odd), each the mean of its bit's sign over the pictures the capture holds, 0.98 of it, with
// ai: noise 0.018 (a LIZARD-512 reading's own over half the blocks) and 0.02 more the deeper the mix, 0.04 while a
// ai: focus hunts, 0.3 on a capture short for the hand or anyway. Two models of the mix (`phasesim.rmodel` = ramp |
// ai: rows; unset, the box display's is the ramp's and the measured one's the rows'):
// ai:   ramp: the change is one instant and the window `winNs` long, f the share of the window after it;
// ai:   rows (the plan in STATUS "Flashing pilots": the two rasters): row u of the picture is exposed from ts + rho u
// ai:     for E (the S26's 8.33 ms, longer by what the case's window is over 9), the display's raster reaches it
// ai:     D u after the change, and each row's mix is its own window's share after the change; a reading is the
// ai:     rows' mean, |r| a curve with one top a refresh (the S26's arithmetic: rho 4.5 ms of readout over the picture, the
// ai:     band where rows mix twice the case's window, since a capture reads its blocks through a light mix). The
// ai:     top sits at the middle of the phases the blocks model reads, so both models agree on where to hold.
// ai: `settle`: the camera's first 4 s as a focus that hunts, a clean capture's blocks from 35% of
// ai: their level rising to all of it and scattered by 6 more (the pilots' r, being a ratio, only noisier, 0.03).
// ai: Which refreshes the sender misses (measured) is drawn a refresh at a time from its own seed, so a run with the
// ai: pilots and one without see the same display (2026-09-30; the shared generator before, the tables moved a little).
// ai: The recorded frames (searches.txt): nine searches of a tripod run, replayed through the lock's own
// ai: fold; each must tell the display's period within what the hold's moves can follow.
// ai:   lizard-android/tools/phasesim/run.sh        (the Kotlin compiler of the Gradle cache; no device)
// ai:   JAVA_TOOL_OPTIONS="-Dphasesim.trace=1 -Dphase.debug=1" run.sh Track "<case>": a line a capture for the first
// ai:   400 (its phase in the display's refresh, r, blocks, the delay asked), and the lock's own debug lines
import dev.lizard.receiver.PhaseLock

// ai: `startNs`: where in the display's refresh the camera starts; `out` (when given) takes the run's share of short
// ai: captures in its first 10 s, from 10 to 30 s and in its last third, its blocks a capture from 10 to 30 s and in
// ai: the last third, and the pace the lock's line names at 10 s and at the end (us a second).
fun run(name: String, mode: PhaseLock.Mode, every: Int, tcNs: Long, tdNs: Double, winNs: Long, secs: Int, quiet: Boolean = false, rough: Boolean = false, measured: Boolean = false, startNs: Long = 0, out: DoubleArray? = null, pilots: Boolean = false, settle: Boolean = false, switchAt: Int = 0, every2: Int = 0): Double {
    val rnd = java.util.Random(7 + startNs)
    val prnd = java.util.Random(13 + startNs)   // ai: the pilots' noise, apart so it moves nothing else
    val lines = ArrayList<String>()
    val p = PhaseLock({ lines.add(it) })
    p.set(mode)
    val rows = (System.getProperty("phasesim.rmodel") ?: (if (measured) "rows" else "ramp")) == "rows"
    // ai: the rows model's geometry (ns): the exposure, the readout over the picture, the raster over it, and the
    // ai: offset that puts the mix's band about the blocks model's (its change is one instant, the band the window
    // ai: before it: the raster is taken as halfway down the picture at that instant)
    val expo = 8_333_333.0 + maxOf(0L, winNs - 9_000_000L)
    val rho = 4_500_000.0
    val delta = 2.0 * winNs - expo
    val rast = rho + delta
    val off = (delta - expo) / 2 + winNs / 2.0
    val NR = 16
    var ts = 1_234_567_890_123L + startNs
    // ai: `switchAt`, `every2`: the painted rate changes at that second of the run to one picture every `every2`
    // ai: refreshes (0: no change), as a sender's rate slider moved mid-run does (2026-10-04)
    val k0 = Math.floor(ts / tdNs).toLong()
    fun everyAt(k: Long) = if (switchAt > 0 && (k - k0) * tdNs >= switchAt * 1e9) every2 else every
    var short10 = 0; var short30 = 0; var blocks30 = 0L; var pace10 = Double.NaN
    val paceOf = { l: String? -> Regex("the pace (-?\\d+) us").find(l ?: "")?.groupValues?.get(1)?.toDouble() ?: Double.NaN }
    val depth = 6
    val asks = ArrayDeque<LongArray>()      // ai: (the frame a delay lands on, its us)
    var delayedPrev = false
    class Cap(val ts: Long, val blocks: Int, val r: Double, val rSd: Double, val r2: Double, val r2Sd: Double)   // ai: the two readings and their own errors
    val all = ArrayList<Cap>()
    var nextStats = ts + 500_000_000L
    // ai: `phasesim.batch=B`: the series as a batching receiver hands it over and a lock asked at every capture: a
    // ai: batch closes B captures on, is decoded 15 ms and 3.2 ms a frame later, and reaches the lock at the next
    // ai: capture (unset: the series asked twice a second, 0.7 s behind, as the app fed it to 2026-10-01).
    // ai: `phasesim.soon=S`: batches of S while the lock asks for its frames soon (`PhaseLock.soon`)
    val batchB = System.getProperty("phasesim.batch")?.toInt() ?: 0
    val soonB = System.getProperty("phasesim.soon")?.toInt() ?: 0
    val halves = System.getProperty("phasesim.halves")?.toInt() ?: 1
    val open = ArrayList<Cap>(); var inBatch = 0
    val due = ArrayDeque<Pair<Long, List<Cap>>>()
    var shortLate = 0; var framesLate = 0; var blocksLate = 0L
    val frames = secs * 60
    // ai: measured: whether the painted frame changes at a refresh (three in four), drawn once a refresh
    val changes = HashMap<Long, Boolean>()
    fun changesAt(k: Long) = k % everyAt(k) == 0L && changes.getOrPut(k) { java.util.Random(k * 1_000_003L + 11 + startNs).nextInt(4) != 0 }
    fun changed(k: Long) = if (measured) changesAt(k) else k % everyAt(k) == 0L
    // ai: the pilots: the refresh last counted and the count of the picture it shows; the count of any refresh from
    // ai: it; the sign a count's bit paints
    var pk = Math.floor(ts / tdNs).toLong()
    var count = 0
    fun countAt(k: Long): Int {
        var p = count
        if (k > pk) { var i = pk + 1; while (i <= k) { if (changed(i)) p++; i++ } }
        else { var i = pk; while (i > k) { if (changed(i)) p--; i-- } }
        return p and 3
    }
    fun sign(c: Int, bit: Int) = (1 - 2 * ((c shr bit) and 1)).toDouble()
    for (n in 0 until frames) {
        var short = false
        var k = Math.ceil((ts - 1_000_000L) / tdNs).toLong()
        while (k * tdNs < ts + winNs + 1_000_000L) {
            if (k % everyAt(k) == 0L) {
                val inside = k * tdNs >= ts && k * tdNs < ts + winNs
                if (inside && (!rough || (k * tdNs >= ts + 1_000_000L && k * tdNs < ts + winNs - 1_000_000L) || rnd.nextBoolean())) short = true
                if (!inside && rough && rnd.nextBoolean()) short = true
            }
            k++
        }
        val shake = rough && (n / 60) % 23 == 22 && n % 60 < 36
        var junk = shake
        if (rough && (rnd.nextInt(100) < 3 || shake)) { short = true; junk = true }
        var blocks = if (short) 3 else 60
        if (measured) {
            // ai: the first change at or after the capture's start, and the last before it
            var c = Math.ceil(ts / tdNs).toLong()
            while (!changesAt(c)) c++
            var b = Math.ceil(ts / tdNs).toLong() - 1
            while (!changesAt(b)) b--
            val wide = tdNs * everyAt(b) - winNs
            val since = ts - b * tdNs
            var mean = if (since > wide + tdNs / 2) 54.0 else 50 - 9 * Math.pow(((since - wide / 2) / (wide / 2)).coerceIn(-1.0, 1.0), 2.0)
            if (c * tdNs < ts + winNs) {
                // ai: a change in the window: read through only near its ends
                val a = minOf(c * tdNs - ts, ts + winNs - c * tdNs)
                mean = if (rnd.nextDouble() < 1 - a / 600_000.0) 41.0 else 2.0
            }
            val broken = shake || rnd.nextInt(1000) < 25
            if (broken) junk = true
            blocks = if (broken) 3 else Math.round(mean + (if (mean > 10) 4 else 2) * rnd.nextGaussian()).toInt().coerceIn(0, 64)
            short = blocks < 32
        }
        // ai: a focus that hunts: the blocks, not whether the capture held a change (short stays the timing's)
        val hunting = settle && n < 240
        if (hunting && blocks > 3) blocks = Math.round(blocks * (0.35 + 0.65 * n / 240) + 6 * prnd.nextGaussian()).toInt().coerceIn(0, 64)
        var r = Double.NaN; var rSd = Double.NaN; var r2 = Double.NaN; var r2Sd = Double.NaN; var meanR = 0.0; var meanR2 = 0.0
        if (pilots) {
            while ((pk + 1) * tdNs <= ts) { pk++; if (changed(pk)) count = (count + 1) and 3 }
            val noise = if (junk) 0.3 else if (hunting) 0.04 else 0.018
            val mean = DoubleArray(2)
            if (rows) {
                for (j in 0 until NR) {
                    val u = (j + 0.5) / NR
                    val w0 = ts + rho * u; val w1 = w0 + expo
                    // ai: the last change to reach the row at or before its exposure began, and the first after
                    var k = Math.floor((w0 + off - rast * u) / tdNs).toLong()
                    while (!changed(k)) k--
                    var c = k + 1
                    while (!changed(c)) c++
                    val tc = c * tdNs + rast * u - off
                    val was = countAt(k); val f = if (tc < w1) (w1 - tc) / expo else 0.0
                    for (bit in 0..1) mean[bit] += (sign(was, bit) * (1 - f) + sign(was + 1, bit) * f) / NR
                }
            } else {
                var c = pk + 1
                while (c * tdNs < ts + winNs && !changed(c)) c++
                val f = if (c * tdNs < ts + winNs) (ts + winNs - c * tdNs) / winNs else 0.0
                for (bit in 0..1) mean[bit] = sign(count, bit) * (1 - f) + sign(count + 1, bit) * f
            }
            rSd = noise + 0.02 * (1 - Math.abs(mean[0])); r2Sd = noise + 0.02 * (1 - Math.abs(mean[1]))
            r = 0.98 * mean[0] + rSd * prnd.nextGaussian(); r2 = 0.98 * mean[1] + r2Sd * prnd.nextGaussian()
            meanR = mean[0]; meanR2 = mean[1]
        }
        // ai: one capture in six is not in the measured series. `phasesim.halves=2`: the app's 2:1 crop, two frames a
        // ai: capture (the receiver's halves: the capture's timestamp and mix, each half its own reading's noise),
        // ai: batched as frames are
        if (!measured || rnd.nextInt(6) != 0) for (h in 0 until halves) {
            val hr = if (h == 0 || !pilots) r else 0.98 * meanR + rSd * prnd.nextGaussian()
            val hr2 = if (h == 0 || !pilots) r2 else 0.98 * meanR2 + r2Sd * prnd.nextGaussian()
            all.add(Cap(ts, blocks, hr, rSd, hr2, r2Sd)); open.add(all.last())
        }
        inBatch += halves
        if (batchB > 0 && inBatch >= (if (soonB > 0 && p.soon) soonB else batchB)) {
            due.addLast(Pair(ts + tcNs + 15_000_000L + 3_200_000L * inBatch, open.toList())); open.clear(); inBatch = 0
        }
        if (n > frames * 2 / 3) { framesLate++; blocksLate += blocks; if (short) shortLate++ }
        if (n < 600 && short) short10++
        if (n in 600 until 1800) { blocks30 += blocks; if (short) short30++ }
        if (n == 600) pace10 = paceOf(p.line(ts))
        var delayed = false
        var dur = tcNs
        if (asks.isNotEmpty() && asks.first()[0] <= n) { val us = asks.removeFirst()[1]; dur += if (measured) (us * 1000 / 5177.5).toLong() * 51775 / 10 else us * 1000; delayed = true }
        val us = p.onCapture(ts, delayedPrev)
        if (System.getProperty("phasesim.trace") == "1" && n < 400) println("  trace %3d ts %.3f s phase %6.3f ms r %6.3f r2 %6.3f blocks %2d %s asked %d us".format(n, (ts - 1_234_567_890_123L) / 1e9, ((ts + off) % tdNs) / 1e6, r, r2, blocks, if (delayedPrev) "delayed" else "       ", us))
        if (us > 0) asks.addLast(longArrayOf(maxOf(n + depth + 1L, (asks.lastOrNull()?.get(0) ?: -1L) + 1), us))
        delayedPrev = delayed
        ts += dur
        if (batchB > 0) {
            while (due.isNotEmpty() && due.first().first <= ts) p.onWindow(due.removeFirst().second.map { PhaseLock.Frame(it.ts, it.blocks, true, it.r, it.rSd, it.r2, it.r2Sd) }, 64)
        } else if (ts >= nextStats) {
            p.onWindow(all.filter { it.ts < ts - 700_000_000L }.takeLast(150).map { PhaseLock.Frame(it.ts, it.blocks, true, it.r, it.rSd, it.r2, it.r2Sd) }, 64)
            nextStats += 500_000_000L
        }
    }
    val share = shortLate.toDouble() / framesLate
    if (out != null) {
        out[0] = short10 / 600.0; out[1] = short30 / 1200.0; out[2] = share; out[3] = blocks30 / 1200.0; out[4] = blocksLate.toDouble() / framesLate
        out[5] = pace10; out[6] = paceOf(p.line(ts))
        return share
    }
    if (!quiet) { println("== $name"); lines.forEach { println("  $it") } }
    println("$name: short in the last third %.1f%% (%d of %d), %.1f blocks a capture; %s".format(100 * share, shortLate, framesLate, blocksLate.toDouble() / framesLate, p.line(ts) ?: "off"))
    return share
}

// ai: Every case from sixteen places in the display's refresh (the starts are what differ), for one mode: a row a
// ai: case, both capture models. `only`: the lock's own lines for the cases whose name holds it, the first eight
// ai: starts. Track is judged (the count of failures returned): where a phase reads, the last third at most 5%
// ai: short, the first 10 s at most 50% and 10 to 30 s at most 10%; with a rough hand on the 9 ms window, the
// ai: last third at most 10%.
fun trial(mode: PhaseLock.Mode, only: String): Int {
    val tc = 16_650_000L
    val hz = { f: Double -> 1e9 / f }
    data class C(val name: String, val every: Int, val tc: Long, val td: Double, val win: Long, val secs: Int, val rough: Boolean = false, val settle: Boolean = false, val switchAt: Int = 0, val every2: Int = 0)
    val cases = listOf(
        C("60.00 Hz, 60 painted, window 9 ms", 1, tc, hz(60.0), 9_000_000L, 300),
        C("60.00 Hz, 60 painted, window 11 ms", 1, tc, hz(60.0), 11_000_000L, 300),
        C("59.94 Hz, 60 painted, window 9 ms", 1, tc, hz(59.94), 9_000_000L, 300),
        C("60.03 Hz, 60 painted, window 9 ms", 1, tc, hz(60.03), 9_000_000L, 300),
        C("a display at the camera's own rate, window 9 ms", 1, tc, tc.toDouble(), 9_000_000L, 300),
        C("60.00 Hz, 30 painted, window 9 ms", 2, tc, hz(60.0), 9_000_000L, 300),
        C("60.12 Hz (the camera the slower), window 9 ms", 1, tc, hz(60.12), 9_000_000L, 300),
        C("60.00 Hz, a camera at 16.700 ms, window 9 ms", 1, 16_700_000L, hz(60.0), 9_000_000L, 300),
        C("rough, 60.00 Hz, window 9 ms", 1, tc, hz(60.0), 9_000_000L, 600, true),
        C("rough, 59.94 Hz, window 9 ms", 1, tc, hz(59.94), 9_000_000L, 600, true),
        C("rough, 60.12 Hz, window 9 ms", 1, tc, hz(60.12), 9_000_000L, 600, true),
        C("60.00 Hz, window 13 ms (3.7 ms read)", 1, tc, hz(60.0), 13_000_000L, 300),
        C("60.00 Hz, window 14.5 ms (2.2 ms read)", 1, tc, hz(60.0), 14_500_000L, 300),
        C("rough, 60.00 Hz, window 14.5 ms (2.2 ms read)", 1, tc, hz(60.0), 14_500_000L, 600, true),
        C("a focus that hunts for 4 s, 60.00 Hz, window 9 ms", 1, tc, hz(60.0), 9_000_000L, 300, settle = true),
        C("a focus that hunts for 4 s, 60.03 Hz, window 9 ms", 1, tc, hz(60.03), 9_000_000L, 300, settle = true),
        C("75 Hz painting 25, window 9 ms", 3, tc, hz(75.0), 9_000_000L, 300),
        C("60.00 Hz, window 17 ms (no phase reads)", 1, tc, hz(60.0), 17_000_000L, 120),
        // ai: the sender's rate slider moved from 30 to 60 mid-run (the 22:02 replay of 2026-10-04): the equal-leak
        // ai: point at 30 painted is the 60-painted optimum, so the hold should stand through the switch
        C("60.00 Hz, 30 painted then 60 at 30 s, window 13 ms (3.7 ms read)", 2, tc, hz(60.0), 13_000_000L, 60, switchAt = 30, every2 = 1),
        C("60.00 Hz, 30 painted then 60 at 30 s, window 17 ms (no phase reads)", 2, tc, hz(60.0), 17_000_000L, 60, switchAt = 30, every2 = 1))
    var fail = 0
    println("%-68s | short, first 10 s: mean, worst | 10 to 30 s: mean, worst | last third: mean, worst | blocks a capture, 10 to 30 s, last third | pace at 10 s: mean, furthest from the end's".format(mode.toString()))
    for (measured in listOf(false, true)) for (c in cases) for (pilots in listOf(false, true)) {
        val name = "${if (pilots) "pilots, " else ""}${if (measured) "measured, " else ""}${c.name}"
        if (only.isNotEmpty() && !name.contains(only)) continue
        if (only.isNotEmpty()) { for (k in 0 until 8) run("$mode, $name, start $k", mode, c.every, c.tc, c.td, c.win, c.secs, false, c.rough, measured, k * c.tc / 8, null, pilots, c.settle, c.switchAt, c.every2); continue }
        val r = (0 until 16).map { k -> DoubleArray(7).also { run(name, mode, c.every, c.tc, c.td, c.win, c.secs, true, c.rough, measured, k * c.tc / 16, it, pilots, c.settle, c.switchAt, c.every2) } }
        fun mean(i: Int) = r.map { it[i] }.filter { !it.isNaN() }.average()
        fun worst(i: Int) = r.maxOf { it[i] }
        println("%-68s | %5.1f%% %5.1f%% | %5.1f%% %5.1f%% | %5.1f%% %5.1f%% | %4.1f %4.1f | %5.0f %5.0f".format(name, 100 * mean(0), 100 * worst(0), 100 * mean(1), 100 * worst(1), 100 * mean(2), 100 * worst(2),
            mean(3), mean(4), mean(5), r.maxOf { Math.abs(it[5] - it[6]) }))
        if (mode != PhaseLock.Mode.Track || c.name.contains("75 Hz") || c.name.contains("no phase reads")) continue
        val bad = if (!c.rough) mean(2) > 0.05 || mean(0) > 0.50 || mean(1) > 0.10 else c.win == 9_000_000L && mean(2) > 0.10
        if (bad) { fail++; println("  FAIL") }
    }
    return fail
}

// ai: The phone's recorded searches through the lock's own fold, a batch of frames at a time as the receiver hands
// ai: them over: when each first tells the display's period, and how far that is from the period the whole run
// ai: folds to. Judged: each tells it, within DRIFT_ERR (what the hold's moves can follow).
fun replay(file: String): Int {
    val ref = 16_649_800L
    var truth = 0.0
    val searches = ArrayList<Pair<String, ArrayList<LongArray>>>()
    for (l in java.io.File(file).readLines()) {
        if (l.startsWith("#") || l.isBlank()) continue
        if (l.startsWith("T ")) { truth = l.substring(2).toDouble() / ref - 1; continue }
        if (l.startsWith("S ")) { searches.add(Pair(l.substring(2), ArrayList())); continue }
        val f = l.split(" ")
        searches.last().second.add(longArrayOf(f[0].toLong(), f[1].toLong()))
    }
    var fail = 0
    println("the S26's recorded searches (the display's period is the camera's frame and %.0f us a second):".format(truth * 1e6))
    for ((name, fs) in searches) {
        var said = false
        var n = 60
        while (!said && n <= fs.size + 29) {
            val m = minOf(n, fs.size)
            val told = PhaseLock.Fold(LongArray(m) { fs[it][0] }, IntArray(m) { fs[it][1].toInt() }, ref, 64).scan()
            if (told.told) {
                said = true
                val off = told.pace - truth
                val bad = Math.abs(off) > PhaseLock.DRIFT_ERR
                println("  %s: told after %3d frames, %.1f s: %5.0f us a second (to %3.0f), %+4.0f from the run's%s".format(name, m, (fs[m - 1][0] - fs[0][0]) / 1e9, told.pace * 1e6, told.err * 1e6, off * 1e6, if (bad) "  FAIL" else ""))
                if (bad) fail++
            }
            n += 30
        }
        if (!said) {
            fail++
            val last = PhaseLock.Fold(LongArray(fs.size) { fs[it][0] }, IntArray(fs.size) { fs[it][1].toInt() }, ref, 64).scan()
            println("  %s: not told in its %d frames (at the end: %.0f us a second, to %.0f)  FAIL".format(name, fs.size, last.pace * 1e6, last.err * 1e6))
        }
    }
    return fail
}

// ai: A recorded series through the lock itself (scripts/exp/replay_phase.py --series: a line `V <version>`, then a
// ai: row a half-frame, ts_ns verified found r rsd r2 r2sd), open loop: the captures in order, each one's rows handed
// ai: over as a batching receiver hands them (the rule above: 32 captures a batch, 8 while the lock asks for them
// ai: soon, read at the first capture after the batch's close plus a frame, 15 ms and 3.2 ms a frame), then onCapture
// ai: with the capture's own timestamp (delayed where its interval is over 16.8 ms; the lock's delays move nothing
// ai: here). The lock's lines as they come (its leaks and moves too under -Dphase.debug=1), each at the capture's
// ai: second from the first, then what it asked over the run.
fun series(file: String) {
    var version = 64
    val rows = ArrayList<PhaseLock.Frame>()
    val vers = ArrayList<Int>()   // ai: the word's version at each row: a `V` line holds from where it stands (2026-10-07; the last one held for the whole run before)
    for (l in java.io.File(file).readLines()) {
        if (l.isBlank() || l.startsWith("#")) continue
        if (l.startsWith("V ")) { version = l.substring(2).trim().toInt(); continue }
        val f = l.trim().split(Regex("\\s+"))
        rows.add(PhaseLock.Frame(f[0].toLong(), f[1].toInt(), f[2] == "1", f[3].toDouble(), f[4].toDouble(), f[5].toDouble(), f[6].toDouble()))
        vers.add(version)
    }
    if (rows.isEmpty()) { println("no rows in $file"); return }
    val t0 = rows.first().ts
    var tNow = t0
    var stands = 0; var searches = 0
    val p = PhaseLock({ println("%7.2f s  %s".format((tNow - t0) / 1e9, it)); if (it.contains("the hold stands")) stands++; if (it.contains("search")) searches++ })
    p.set(PhaseLock.Mode.Track)
    val open = ArrayList<PhaseLock.Frame>()
    val due = ArrayDeque<Triple<Long, List<PhaseLock.Frame>, Int>>()
    val asks = ArrayList<Long>()
    var i = 0; var prevTs = 0L; var captures = 0; var inBatch = 0
    while (i < rows.size) {
        val ts = rows[i].ts
        while (i < rows.size && rows[i].ts == ts) { open.add(rows[i]); i++ }
        captures++; inBatch++
        tNow = ts
        while (due.isNotEmpty() && due.first().first <= ts) { val d = due.removeFirst(); p.onWindow(d.second, d.third) }
        if (inBatch >= (if (p.soon) 8 else 32)) { due.addLast(Triple(ts + 16_650_000L + 15_000_000L + 3_200_000L * open.size, open.toList(), vers[i - 1])); open.clear(); inBatch = 0 }
        val delayed = prevTs != 0L && ts - prevTs > 16_800_000L
        val us = p.onCapture(ts, delayed)
        if (us > 0) asks.add(us)
        prevTs = ts
    }
    println("%7.2f s  %s".format((tNow - t0) / 1e9, p.line(tNow) ?: "off"))
    println("%d captures over %.1f s: %d delays asked, %.2f ms in all (the largest %.3f), %d stands, %d search lines".format(
        captures, (tNow - t0) / 1e9, asks.size, asks.sum() / 1e3, (asks.maxOrNull() ?: 0L) / 1e3, stands, searches))
}

fun main(args: Array<String>) {
    // ai:   run.sh                      the first model's checks, the recorded searches, then track's table, judged
    // ai:   run.sh Track[,Scan] [text]  a mode's table, or the lock's lines for the cases named
    // ai:   run.sh replay               the recorded searches only
    // ai:   run.sh series <file>        a recorded series (replay_phase.py --series) through the lock, open loop
    val data = System.getProperty("phasesim.dir", ".") + "/searches.txt"
    if (args.isNotEmpty() && args[0] == "replay") { System.exit(if (replay(data) == 0) 0 else 1) }
    if (args.isNotEmpty() && args[0] == "series") { series(args.getOrElse(1) { "" }); return }
    if (args.isNotEmpty()) { for (m in args[0].split(",")) trial(PhaseLock.Mode.valueOf(m), args.getOrElse(1) { "" }); return }
    val tc = 16_650_000L                    // the S26's frame at "60 a second"
    val hz = { f: Double -> 1e9 / f }        // a display's refresh, ns
    var fail = 0
    fun want(share: Double, most: Double) { if (share > most) { fail++; println("  FAIL: over %.0f%%".format(100 * most)) } }
    // ai: `every`: refreshes a painted frame (1: 60 painted on a 60 Hz display; 2: 30 painted)
    run("off, 60.00 Hz, 60 painted, window 9 ms", PhaseLock.Mode.Off, 1, tc, hz(60.0), 9_000_000L, 120, true)
    want(run("scan (a 60 Hz grid), 60.00 Hz, 60 painted, window 9 ms", PhaseLock.Mode.Scan, 1, tc, hz(60.0), 9_000_000L, 120, true), 0.0)
    run("off, rough, 60.00 Hz, 60 painted, window 9 ms", PhaseLock.Mode.Off, 1, tc, hz(60.0), 9_000_000L, 600, true, true)
    // ai: free-running where the phases that read are narrow (slow pixels)
    run("off, 60.00 Hz, 60 painted, window 13 ms (3.7 ms read)", PhaseLock.Mode.Off, 1, tc, hz(60.0), 13_000_000L, 300, true)
    run("off, 60.00 Hz, 60 painted, window 14.5 ms (2.2 ms read)", PhaseLock.Mode.Off, 1, tc, hz(60.0), 14_500_000L, 300, true)
    fail += replay(data)
    fail += trial(PhaseLock.Mode.Track, "")
    println(if (fail == 0) "ok" else "FAILED: $fail")
    if (fail != 0) System.exit(1)
}
