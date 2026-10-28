package dev.lizard.phase.d6

// ai: The experimental arm D6 of the pilots-only, self-healing phase lock (2026-10-27: Receive's Experimental fold,
// ai: Settings.d6 "on"; the app's lock otherwise is ../PhaseLock.kt). Engine feeds it as it feeds that one, tells
// ai: it what became of each delay request (delayFate) and gives it no readout skew and no code row.

import kotlin.math.abs

// ai: The camera's phase against the display (2026-09-30).
// ai: A capture that holds a change of the painted frame is two symbols and reads little, and whether captures hold
// ai: one depends on where they start in the display's refresh. A camera and a display each keep their own clock, so
// ai: that place slides: the S26's frame is 16.650 ms, 1.1 ms a second against a 60.00 Hz display, every phase in
// ai: 16 s. One frame made longer moves every frame after it (Engine.delay). That is the only lever: a delay of x
// ai: moves the phase x later, and a delay of a whole period less x moves it x earlier (a capture skipped).
// ai:
// ai: Track holds by the pilots (research/SPEC.md 7.3; signed since 2026-10-01, STATUS "Signed pilots"). The sender
// ai: paints its count of pictures mod 4 into the block tails, bit 0 on the even blocks and bit 1 on the odd, and a
// ai: capture reads r (the even blocks) and r2 (the odd): their signs the count of the picture it mostly holds, and
// ai: their sizes what it holds of the pictures either side, each neighbour lowering the reading whose bit it does not
// ai: share. Every frame of the format carries both readings (2026-10-25: one block a frame, which had bit 0 alone,
// ai: left the format), and a frame without them is no evidence. So the hold needs no search and no sweep:
// ai:   the hold begins where the camera is (`here`), on the first frames;
// ai:   the leaks (`leaks`): over the frames since the last move, each reading is K less twice K times the share
// ai:     of each neighbour whose bit differs (which neighbours do is read off the neighbouring captures' counts,
// ai:     so a picture the screen showed twice or skipped is counted for what it was, not guessed): K, the share
// ai:     a of the picture before and b of the one after by least squares, each reading by its own error;
// ai:   the move: a capture that starts too early holds some of the picture before, one too late some of the
// ai:     next, so the line goes later by `gain` ms a unit of a - b wherever that is Z of its errors from 0; the
// ai:     gain is measured by the moves (what the captures' places moved over what it changed a - b by);
// ai:   deep in a mix (no capture mostly one picture, so no count): a quarter of a refresh later, and look again;
// ai:   the pace: from pairs of readings (`learn`) and from a flat top's edge reached again.
// ai: Nothing it learns is ever certain (2026-10-26). The pace and the gain are one estimate with its covariance
// ai: (`observe`): every pair of readings and every edge reached again is one look at them, with the variance its
// ai: readings, the gain's own error, the view's wander over its span (a hand's, re-aims and all: VIEW_MS2) and, over what
// ai: of it was the AIM_NS after the session's first hold began (the code in view), a hand still aiming (AIM_MS2) give it; a look beyond Z of what is
// ai: expected moves them only as far as Z (the view jumps where a hand aims; a jump is the view's, not the
// ai: clocks'); both drift by their own process noise between looks (GAIN_Q: the window over the code moves with a
// ai: hand; PACE_Q: the slide the hold sees is the clocks' and a hand's slow turn, which changes within seconds). The newest N_EVID looks (of the newest LONG_NS) re-check the pace:
// ai: where their median (a vote each, at the gain as it is) is beyond Z of its error and the pace's own, the pace is
// ai: that median with its error (a bad start, a view's jump taken for a slide, a kept pace another screen refutes),
// ai: and the gain goes back to what it was before those looks (they told it at a pace that was wrong). So the pace a reopen keeps is a
// ai: prior the new session's looks confirm or overturn, and what the log claims is the estimate's own error.
// ai: A capture's place is the time its code is read: the camera reads its rows down the sensor over its readout
// ai: skew (Camera2's SENSOR_ROLLING_SHUTTER_SKEW, its first row to one past its last) and stamps a capture with its
// ai: first row's (SENSOR_TIMESTAMP), so the code's mean row (the receiver's quad, a frame of the series each, as a
// ai: share of the sensor's rows) is read at the timestamp plus the skew times that share. A code that moves up or
// ai: down the picture (a turn of the phone, an aim moved) moves where it is read and so the mix; held on its read
// ai: time, the camera follows it by delays, and no balance moves and no slide is left for the pace to learn.
// ai: Self-healing (2026-10-25): what the hold knows belongs to what it describes, and goes when that does.
// ai:   The camera's (its frame, the delays' landings, the session's first and newest capture, the newest frame
// ai:     counted): from its own timestamps whatever the mode; afresh for a new session (`reopened`), never at a set.
// ai:   The clocks' (the display's refresh on the camera's clock and how well it is known): the state the line steps
// ai:     by, apart from the camera's frame, which only says how many refreshes a capture spans; kept through a
// ai:     reopen, afresh at a set of the mode; a frame that is no whole number of refreshes is another display's.
// ai:   The view's (the gain and its fit, a flat top's width, the code's row) and the hold's place: afresh at a
// ai:     reopen, and the place where the camera's frame changes; a flat top's width measured again once the hold has
// ai:     stood clear of its edges for LONG_NS, and the gain bounded by a step that crossed the top.
// ai: A delay lands where the timestamps show it (an interval longer than the camera's frame by over half of it).
// ai: What became of the request the host says (`delayFate`: the camera took it, or it never will: refused, failed
// ai: before its frame, its sequence aborted), so a delay the camera did not take is known, not waited for. One the
// ai: camera took whose interval has not shown is given up once its mark is further behind than the camera's own
// ai: newest landings were behind theirs, and one more capture (a frame's duration sets the interval to the frame
// ai: after it). Of a delay nothing is heard of, the camera's clock past its ask by LOST frames (taken at the newest
// ai: time the camera is known to have reached, so a backlog after a stall of the engine's thread gives up nothing on
// ai: its way) gives it up: Camera2 promises one or the other for every request, so only a host that breaks that
// ai: leaves one there.
// ai: Every call is on the engine's thread. Plain JVM (liblizard/kotlin); its one host is the Android app's Engine
// ai: (a Camera2 lens with per-frame control: Engine.lockBarOf).
class PhaseLock(private val say: (String) -> Unit) {
    enum class Mode { Off, Track }
    // ai: r, rSd: the pilots' reading over the even blocks and its standard error; r2, r2Sd: over the odd; row: the
    // ai: code's mean row as a share of the sensor's rows, in the order the camera reads them (NaN: not known)
    class Frame(val ts: Long, val blocks: Int, val found: Boolean, val r: Double = Double.NaN, val rSd: Double = Double.NaN,
                val r2: Double = Double.NaN, val r2Sd: Double = Double.NaN, val row: Double = Double.NaN)
    // ai: The lock's state for the stats row and a replay's meta (Engine.phaseState, MainActivity's poll; 2026-10-04): a
    // ai: snapshot made on the engine's thread at each decision, read from any. `arm`: pilots; `what`: the word line()
    // ai: uses (holds, no frames yet); the newest estimate of the leaks (a and b the shares of the pictures
    // ai: before and after in a capture, k what a capture of one picture reads, se the error of a - b, n its frames;
    // ai: NaN with none); the gain (ms a unit of a - b) and the pace (us a second); whether the hold stands; the edge
    // ai: of a flat top it stepped in from (1 early, -1 late, 0 none) and the top's width (ms); the delays asked since
    // ai: the mode was set; the display's refresh on the camera's clock (ms, NaN before a hold), the refreshes a
    // ai: capture spans, the camera's frame (ms); the longest landing of a delay in the newest LONG_NS (ms from its
    // ai: ask, NaN none); the code's row.
    class State(val arm: String, val what: String, val a: Double, val b: Double, val k: Double, val se: Double, val n: Int,
                val gain: Double, val pace: Double, val stood: Boolean, val edge: Int, val flatMs: Double, val delays: Int,
                val refreshMs: Double, val multiple: Int, val frameMs: Double, val landMs: Double, val row: Double)
    @Volatile var state: State? = null
        private set

    var mode = Mode.Off
        private set
    private var delays = 0
    private var delaysAll = 0       // ai: delays asked since the mode was set (State)

    // ai: the camera's: the session's first capture and its newest (frames captured outside them are another
    // ai: session's), the newest frame of the series counted, whether the newest capture was a delayed one, its
    // ai: readout skew (ns, the sensor's first row to one past its last); the newest RECENT_NS of intervals between two frames
    // ai: not delayed (end, length) and their median, the camera's frame
    private var first = 0L
    private var now = 0L
    private var seen = 0L
    private var wasDelayed = false
    private var skew = 0L
    private val plains = ArrayDeque<LongArray>()
    private var frame = 0L
    // ai: the delay out (ns; 0: none), whether its frame was marked delayed (or the camera said it took it) and the
    // ai: captures since, whether the capture before ended an interval that long, the camera time it was asked at; and
    // ai: the newest LONG_NS of landings (when, ns from the ask, captures from the mark)
    private var owed = 0L
    private var flagged = false
    private var flagAt = 0
    private var longPrev = false
    private var askT = 0L
    private val lands = ArrayDeque<LongArray>()
    // ai: the clocks': the display's refresh on the camera's clock is base (1 + pace) (base: the camera's frame when
    // ai: the refresh was last taken from it; 0: not yet), how well the pace is known (ns a ns; MAX: not at all), and
    // ai: the whole number of refreshes in the camera's frame, which the line steps
    private var base = 0L
    private var pace = 0.0
    private var paceVar = PACE_VAR0   // ai: (ns a ns)²; PACE_VAR0: not measured (as far off as two clocks may be)
    private var paceAt = 0L           // ai: the camera time the estimate's process noise was last brought to
    private var multiple = 1
    private val sure get() = Math.sqrt(paceVar)
    private val refresh get() = base * (1 + pace)
    private val period get() = refresh * multiple
    // ai: the hold
    private var row = ROW           // ai: the code's row, the newest frame's that has one
    private var settle = 0L         // ai: frames after this one were captured after the last move landed
    private var line = -1L          // ai: the read time of a capture on the phase held, the one nearest the newest capture; -1: none
    private var move = false        // ai: the line was moved: the camera follows at once
    private var moving = false      // ai: the delay that follows it is on its way
    private val window = ArrayDeque<Frame>()  // ai: the hold's frames, the newest LONG_NS
    private var since = 0L          // ai: frames captured after this are the hold's
    private var level = 0.0         // ai: blocks a frame (the log's)
    // ai: each capture's timestamp against the line when it was taken (ts - line, ns) and its skew, the newest LONG_NS:
    // ai: a frame's place is that and the skew times its own row
    private val placed = java.util.TreeMap<Long, LongArray>()
    // ai: the pilots' fit: ms the line moves a unit of a - b. The reading before (a - b, its error, its frames' mean
    // ai: time and place against the line; 0: none that showed a neighbour), the move made on it (ms) and the pace it
    // ai: was made at; and the sums of the fit of the gain and the display's pace to the readings' history (`learn`).
    private var gain = GAIN_MS
    private var gainVar = GAIN_VAR0   // ai: (ms a unit)²
    private var gpCov = 0.0           // ai: the gain's and the pace's covariance (ms a unit, ns a ns)
    private var prevE = 0.0
    private var prevSe = 0.0
    private var prevT = 0L
    private var prevHop = 0.0
    private var prevPace = 0.0
    private var prevX = 0.0
    private var prevSide = 0        // ai: which neighbours it showed: 1 the one before, 2 the one after, 3 both
    // ai: the session's newest N_EVID looks at the gain and the pace (z ms, its gain's and pace's coefficients, its
    // ai: variance ms²) and how many the session has had
    private val evid = ArrayDeque<DoubleArray>()
    private var aimFrom = 0L        // ai: the session's first hold began (the code in view): a hand may still aim for AIM_NS
    private var balanced = false    // ai: this hold has said it stands (a line for the log, once a hold)
    private var lastLeak: Leak? = null   // ai: the newest estimate of the leaks (State, line)
    // ai: a top that is flat (a capture there holds neither neighbour): the edge the hold last stepped in from
    // ai: (1: the early edge, so the step was later; -1: the late; 0: none), how far in it stepped (ns) and the
    // ai: frame it did so at; half the top's width where a step in from one edge met the other (0: not known)
    private var edgeDir = 0
    private var edgeIn = 0L
    private var edgeT = 0L
    private var edgeClean = false   // ai: frames since that step have held neither neighbour (it landed inside the top)
    private var flatHalf = 0L
    // ai: whether the hold has asked no move (State.stood), until a mix too deep to read or a new hold
    private var stood = false

    // ai: Track starts its hold and the clocks afresh, Off stops it; what the camera's timestamps have measured
    // ai: stands either way (the intervals are the camera's, not the mode's).
    fun set(m: Mode) {
        if (m == mode) return
        mode = m
        base = 0; pace = 0.0; paceVar = PACE_VAR0; paceAt = 0; multiple = 1
        hold(); delaysAll = 0
        say(if (m == Mode.Track) "phase: track" else "phase: off")
    }

    // ai: The camera was opened again (a setting changed, the app came back, a logical camera switched its lens): a
    // ai: new session, its frames starting anywhere in the display's refresh and maybe on a clock restarted below the
    // ai: old one; the display's refresh is the display's and stands, and the view's and the hold's are learned afresh.
    fun reopened() {
        first = 0; now = 0; seen = 0; wasDelayed = false; skew = 0; plains.clear(); frame = 0
        owed = 0; flagged = false; longPrev = false; askT = 0; lands.clear()
        hold()
    }

    private fun hold() {
        row = ROW; settle = 0; line = -1; move = false; moving = false; window.clear(); since = 0; level = 0.0; placed.clear()
        gain = GAIN_MS; gainVar = GAIN_VAR0; gpCov = 0.0; evid.clear(); aimFrom = 0
        prevT = 0; balanced = false; lastLeak = null
        edgeDir = 0; flatHalf = 0; stood = false; state = null
    }

    // ai: A capture's result: its timestamp, whether it was a delayed frame, and the camera's readout skew over the
    // ai: sensor's rows (0 where the host has none). Returns the microseconds to delay the next frame by, 0 for none.
    fun onCapture(tsNs: Long, delayed: Boolean, skewNs: Long = 0L): Long {
        if (tsNs == 0L) return 0
        if (now == 0L) { first = tsNs; settle = tsNs - 1 }
        val dt = if (now != 0L) tsNs - now else 0L
        val plain = dt > 0 && !delayed && !wasDelayed
        // ai: the camera's frame is the median of the newest RECENT_NS of intervals, and it is that only once the
        // ai: session has run that long (a session's first intervals are often irregular, and a median of one is the
        // ai: one, a dropped frame among them): until then a session that kept the clocks takes their period for it and
        // ai: nothing is taken from it; a cold start, with nothing kept, begins from what is in and is put right then
        val known = tsNs - first >= RECENT_NS
        if (plain) {
            plains.addLast(longArrayOf(tsNs, dt))
            while (plains.first()[0] < tsNs - RECENT_NS) plains.removeFirst()
            val d = LongArray(plains.size) { plains[it][1] }; d.sort()
            frame = if (!known && base != 0L) Math.round(period) else d[(d.size - 1) / 2]
        }
        wasDelayed = delayed
        now = tsNs
        skew = maxOf(0L, skewNs)
        landed(tsNs, dt, delayed)
        if (plain && base != 0L && known) steps()
        if (mode == Mode.Off) return 0
        return track(tsNs, delayed)
    }

    // ai: The refreshes a capture spans: the whole number nearest the camera's frame over the refresh. A frame no
    // ai: whole number of refreshes makes to within PACE_MOST (the most two clocks differ) is not this refresh's (one
    // ai: taken from a camera at another rate, or another screen): the refresh taken from the camera's frame afresh.
    // ai: Either way the hold begins again where the camera is: its frames so far were captured a refresh or more apart
    // ai: from what the line steps now, so neither their places nor their neighbours (a capture's neighbours are the
    // ai: pictures either side only where it spans one refresh) say anything of the new hold.
    private fun steps() {
        val m = Math.round(frame / refresh)
        if (m >= 1 && abs(frame / (m * refresh) - 1) <= PACE_MOST) {
            if (m.toInt() == multiple) return
            say("phase: the camera's frame %.3f ms is %d refreshes of %.3f ms".format(frame / 1e6, m, refresh / 1e6))
            multiple = m.toInt()
        } else {
            say("phase: the camera's frame %.3f ms is no whole number of refreshes of %.3f ms: the refresh taken from it".format(frame / 1e6, refresh / 1e6))
            base = frame; pace = 0.0; paceVar = PACE_VAR0; gpCov = 0.0; multiple = 1
            evid.clear()
        }
        if (line >= 0) here()
    }

    // ai: The delay out, whatever the mode: landed at the first interval longer than the camera's frame by over half
    // ai: the delay, from the capture before its marked frame on (a camera that applies the longer frame captures after
    // ai: the mark lands it there; on the S26 at the mark): only frames after it are the move's. Given up (the header
    // ai: says when), and the camera asked onto the line afresh (a pair of readings across it counts what the captures'
    // ai: places show it did, nothing if it never landed).
    private fun landed(ts: Long, dt: Long, delayed: Boolean) {
        if (owed == 0L) return
        val long = dt - frame > owed / 2
        val was = longPrev
        longPrev = long
        if (delayed) { flagged = true; flagAt = 0 }
        if (flagged && (long || (delayed && was))) {
            owed = 0
            lands.addLast(longArrayOf(ts, ts - askT, flagAt.toLong()))
            while (lands.first()[0] < ts - LONG_NS) lands.removeFirst()
            if (moving) { moving = false; settle = ts }
        } else if (if (flagged) ++flagAt > (lands.maxOfOrNull { it[2] }?.toInt() ?: LOST) + 1 else ts - askT > LOST * frame) {
            if (DEBUG) say("phase: DEBUG a delay of %.3f ms not landed %.0f ms after its ask (%s): given up".format(owed / 1e6, (ts - askT) / 1e6, if (flagged) "taken, %d captures on".format(flagAt) else "nothing heard"))
            giveUp()
        }
    }
    private fun giveUp() { owed = 0; if (moving) { moving = false; move = true } }

    // ai: What became of the delay asked, as the host hears it (Engine's capture callback and delay()): taken, its frame
    // ai: captured though its result failed (its interval then shows the delay as a marked frame's would); not taken
    // ai: (delay() refused it, its request failed before its frame, its sequence was aborted): given up at once.
    fun delayFate(taken: Boolean) {
        if (owed == 0L) return
        if (taken) { flagged = true; flagAt = 0 } else giveUp()
    }

    // ai: The receiver's newest frames (ns on the camera's clock, verified blocks, whether a symbol was registered) and
    // ai: the blocks a frame the word names. The receiver's series hands out the current camera stream's frames alone
    // ai: (receiver.cpp's Stamps, each row with the stream its frame came in), which a clock that restarts below the
    // ai: old one needs; a frame captured before this session's first capture is another session's here too.
    fun onWindow(given: List<Frame>, version: Int) {
        if (now == 0L) return
        val frames = captures(given.filter { it.ts >= first })
        val newest = frames.maxOfOrNull { it.ts } ?: return
        if (mode == Mode.Track && version > 0) tracked(frames.filter { it.ts > seen && it.found && !it.r.isNaN() && !it.r2.isNaN() }.sortedBy { it.ts })
        seen = maxOf(seen, newest)
    }

    // ai: One frame a capture. The 2:1 crop gives the receiver two frames a capture (its halves, the capture's
    // ai: timestamp both), and they are one look at one mix of pictures: as two frames, each half's neighbouring
    // ai: capture in `leaks` was its own twin and each capture's mix counted twice as evidence (STATUS "The oscillation
    // ai: in 2:1"). Frames of one timestamp are one: each pilot reading their mean by inverse variance, the blocks
    // ai: and the code's row their mean, found where either was.
    private fun captures(fs: List<Frame>): List<Frame> {
        if (fs.size < 2) return fs
        val out = ArrayList<Frame>(fs.size)
        var i = 0
        while (i < fs.size) {
            var j = i + 1
            while (j < fs.size && fs[j].ts == fs[i].ts) j++
            out.add(if (j - i == 1) fs[i] else merged(fs.subList(i, j)))
            i = j
        }
        return out
    }
    private fun merged(g: List<Frame>): Frame {
        fun mean(v: (Frame) -> Double, sd: (Frame) -> Double): Pair<Double, Double> {
            var w = 0.0; var s = 0.0
            for (f in g) { val x = v(f); if (x.isNaN()) continue; val e = sd(f).let { if (it.isNaN() || it <= 0) R_NOISE_MIN else it }; w += 1 / (e * e); s += x / (e * e) }
            return if (w == 0.0) Pair(Double.NaN, Double.NaN) else Pair(s / w, 1 / Math.sqrt(w))
        }
        val (r, rSd) = mean({ it.r }, { it.rSd }); val (r2, r2Sd) = mean({ it.r2 }, { it.r2Sd })
        val rows = g.filter { !it.row.isNaN() }
        return Frame(g[0].ts, Math.round(g.sumOf { it.blocks }.toDouble() / g.size).toInt(), g.any { it.found }, r, rSd, r2, r2Sd,
            if (rows.isEmpty()) Double.NaN else rows.sumOf { it.row } / rows.size)
    }

    // ai: A capture (track): the delay to ask, us. Where a capture sat comes from its own timestamp and its code's
    // ai: row, never from what was asked of the camera, so a delay that lands late, short or not at all costs nothing
    // ai: but time.
    private fun track(ts: Long, delayed: Boolean): Long {
        if (base == 0L || line < 0) return 0
        // ai: its code's read time, and the line's point nearest it
        val rt = ts + Math.round(skew * row)
        val k = Math.round((rt - line) / period)
        if (k != 0L) line += Math.round(k * period)
        placed[ts] = longArrayOf(ts - line, skew)
        while (placed.firstKey() < ts - LONG_NS) placed.pollFirstEntry()
        // ai: nothing is asked on a delayed frame's own timestamp, nor while a delay is out
        if (owed > 0 || delayed) return 0
        var ns = 0L
        // ai: early: delayed onto the line as soon as it is a band early (a move's, at once). Late (a camera slower
        // ai: than the display, a line moved earlier, a code moved down the picture): a period less what it is late,
        // ai: a capture skipped
        val early = line - rt
        val far = if (move) Math.round(refresh / 256) else TOOTH_NS
        if (early >= minOf(far, BAND_NS)) ns = early else if (early <= -far) ns = Math.round(period) + early
        if (move) { if (ns != 0L) moving = true else settle = ts; move = false }
        if (ns <= 0) return 0
        owed = ns; flagged = false; flagAt = 0; longPrev = false; askT = maxOf(ts, seen); delays++; delaysAll++
        return maxOf(1, ns / 1000)
    }

    // ai: The hold begun where the camera is: the line through the newest capture's read time, so the camera is on it
    // ai: and its frames from then on are the hold's; the clocks as they are (from the camera's frame, where the refresh
    // ai: is not known yet).
    private fun here() {
        if (frame == 0L) return
        if (base == 0L) { base = frame; pace = 0.0; paceVar = PACE_VAR0; gpCov = 0.0; multiple = 1 }
        prevT = 0; edgeDir = 0; flatHalf = 0; balanced = false; stood = false; lastLeak = null; placed.clear()
        if (aimFrom == 0L) aimFrom = now
        line = now + Math.round(skew * row); move = false; moving = false; window.clear(); since = now; settle = now; level = 0.0
        say("phase: the hold begins: the refresh %.4f ms on the camera's clock, %d a frame of %.4f ms, the code at row %.2f".format(refresh / 1e6, multiple, frame / 1e6, row))
    }

    // ai: three equations in three unknowns, m a 3 x 7 array (the matrix, the right side, the identity): solved in
    // ai: place by elimination, the answers m[k][3] / m[k][k] and the inverse's diagonal m[k][4 + k] / m[k][k];
    // ai: false where they do not determine the unknowns
    private fun solve3(m: Array<DoubleArray>): Boolean {
        for (c in 0 until 3) {
            var r = c; for (k in c + 1 until 3) if (abs(m[k][c]) > abs(m[r][c])) r = k
            val tmp = m[c]; m[c] = m[r]; m[r] = tmp
            if (abs(m[c][c]) < 1e-9) return false
            for (k in 0 until 3) if (k != c) { val f = m[k][c] / m[c][c]; for (j in 0..6) m[k][j] -= f * m[c][j] }
        }
        return true
    }

    // ai: What the hold's frames after `from` hold of their neighbours. A frame's count is its two signs (bit 0 from
    // ai: r, bit 1 from r2), and so are the captures' before and after it, a capture either side (a frame whose
    // ai: neighbour is not in the series, or any of the three with a reading under R_FLASH, whose sign may be
    // ai: noise, is left out). Each of its readings is then, in size, K - 2 K a [the bit differs from the count
    // ai: before] - 2 K b [it differs from the count after]: K what a capture of one picture reads, a and b the
    // ai: shares of the pictures before and after. Least squares over both readings of every frame, each by its
    // ai: own error, for K, K a and K b. Null: too few frames, or frames that do not tell the three apart (a
    // ai: sender whose count does not change: no bit ever differs).
    // ai: t: the frames' mean time; k: what a capture of one picture reads; x: the frames' mean place against the line (ns)
    private class Leak(val a: Double, val b: Double, val sa: Double, val sb: Double, val n: Int, val t: Long, val k: Double, val x: Double)
    private fun leaks(from: Long, to: Long = Long.MAX_VALUE): Leak? {
        val fs = window
        val t = period
        val m = Array(3) { DoubleArray(7) }
        for (k in 0 until 3) m[k][4 + k] = 1.0
        var n = 0; var ts = 0.0; var xs = 0.0; var xn = 0
        fun sure(f: Frame) = abs(f.r) >= R_FLASH && abs(f.r2) >= R_FLASH
        fun count(f: Frame) = (if (f.r < 0) 1 else 0) or (if (f.r2 < 0) 2 else 0)
        for (i in 1 until fs.size - 1) {
            val f = fs[i]; val p = fs[i - 1]; val q = fs[i + 1]
            if (f.ts <= from || f.ts > to || f.ts - p.ts > t * 3 / 2 || q.ts - f.ts > t * 3 / 2 || !sure(f) || !sure(p) || !sure(q)) continue
            val c = count(f); val before = count(p) xor c; val after = count(q) xor c
            for (bit in 0..1) {
                val y = abs(if (bit == 0) f.r else f.r2)
                val sd = maxOf(R_NOISE_MIN, (if (bit == 0) f.rSd else f.r2Sd).let { if (it.isNaN()) R_NOISE_MIN else it })
                val w = 1 / (sd * sd)
                val x = doubleArrayOf(1.0, -2.0 * ((before shr bit) and 1), -2.0 * ((after shr bit) and 1))
                for (u in 0 until 3) { for (v in 0 until 3) m[u][v] += w * x[u] * x[v]; m[u][3] += w * x[u] * y }
            }
            n++; ts += (f.ts - from).toDouble()
            // ai: the frame's place as its capture had it, at the frame's own row (the series' times are the captures'
            // ai: to the ns, give or take one)
            placed.floorEntry(f.ts + frame / 2)?.takeIf { it.key > f.ts - frame / 2 }?.let { xs += it.value[0] + it.value[1] * (if (f.row.isNaN()) row else f.row); xn++ }
        }
        if (n < L_NEED || !solve3(m)) return null
        val k = m[0][3] / m[0][0]
        if (k < R_FLASH) return null
        return Leak(m[1][3] / m[1][1] / k, m[2][3] / m[2][2] / k, Math.sqrt(maxOf(0.0, m[1][5] / m[1][1])) / k, Math.sqrt(maxOf(0.0, m[2][6] / m[2][2])) / k, n, from + Math.round(ts / n), k, if (xn > 0) xs / xn else 0.0)
    }

    // ai: The receiver's new frames (track), in capture order: the code's row from the newest that has one, the hold
    // ai: begun on the first, then each batch read.
    private fun tracked(got: List<Frame>) {
        if (got.isEmpty()) return
        got.lastOrNull { !it.row.isNaN() }?.let { row = it.row.coerceIn(0.0, 1.0) }
        if (line < 0) { here(); return }
        for (f in got) if (f.ts > since) window.addLast(f)
        while (window.isNotEmpty() && window.first().ts < window.last().ts - LONG_NS) window.removeFirst()
        decide()
    }

    // ai: What the hold's frames say, once a batch of them is in.
    private fun decide() {
        if (move || moving || window.isEmpty()) return
        held()
        snap()
    }

    // ai: The gain and the display's pace from the hold's own history. Between two readings that both showed a
    // ai: neighbour, the captures went later by h (ms: what their own places against the line show, with the line's
    // ai: move) and earlier by the slide (the display's pace p less the line's, over the time dt), and a - b fell by
    // ai: that over the gain G: G (e2 - e1) - p dt = -h - pace dt, one look at (G, p) with the readings' variance
    // ai: through the gain. The gain is told only by a pair whose move is Z of what its readings resolve (the gain
    // ai: times the error of e2 - e1): a pair chosen by a - b's own change is chosen by its noise, and the gain told by
    // ai: those falls toward GAIN_LEAST the longer a hold lasts; elsewhere the look tells the pace alone.
    private fun learn(t: Long, de: Double, dt: Double, h: Double, paceMs: Double, w: Double) =
        observe(t, -h - paceMs * dt, de, -dt * 1e3, gain * gain / w, dt, h * h * w >= Z * Z * gain * gain)

    // ai: One look: z (ms) = hg gain + hp pace, its readings' variance rRead (ms²) over dt seconds to camera time t.
    // ai: The view's wander over dt (and, over what of it was the session's first AIM_NS, an aiming hand's) is added; a look beyond Z of its expected
    // ai: spread counts as Z (the gain and the pace move only as far as that); `both`: the gain is told too, else the
    // ai: pace alone with the gain's error carried (its covariance kept whole either way).
    private fun observe(t: Long, z: Double, hg: Double, hp: Double, rRead: Double, dt: Double, both: Boolean) {
        age(t)
        // ai: the aim: the part of the look's span within the session's first AIM_NS (a hand still aiming)
        val aim = if (aimFrom == 0L) 0.0 else maxOf(0.0, (minOf(t, aimFrom + AIM_NS) - maxOf(t - Math.round(dt * 1e9), aimFrom)) / 1e9)
        // ai: a look says the pace's mean over its span, which is off the pace now by the pace's own process noise over
        // ai: the span (a random walk's mean over a span T is off its end by a third of its variance over T)
        val span = hp * hp * PACE_Q * dt / 3
        val r = rRead + span + VIEW_MS2 * dt + AIM_MS2 * aim * aim
        evid.addLast(doubleArrayOf(z, hg, hp, r, rRead + span, gain, gainVar, t.toDouble())); while (evid.size > N_EVID) evid.removeFirst()
        val nu = z - hg * gain - hp * pace
        if (DEBUG) say("phase: DEBUG a look over %.1f s (%s): the pace %.0f us a second (to %.0f) at the gain %.1f, against %.0f (to %.0f)".format(dt, if (both) "gain too" else "pace", (z - hg * gain) / hp * 1e6, Math.sqrt((r + hg * hg * gainVar) / (hp * hp)) * 1e6, gain, pace * 1e6, sure * 1e6))
        val pg = gainVar * hg + gpCov * hp; val pp = gpCov * hg + paceVar * hp
        val hph = hg * pg + hp * pp
        val s = maxOf(hph + r, nu * nu / (Z * Z))
        if (both) {
            val kg = pg / s; val kp = pp / s
            gain = (gain + kg * nu).coerceIn(GAIN_LEAST, GAIN_MOST); pace = (pace + kp * nu).coerceIn(-PACE_MOST, PACE_MOST)
            gainVar -= kg * pg; gpCov -= kg * pp; paceVar -= kp * pp
        } else {
            // ai: the pace's row alone (Joseph's form: the gain kept as it is, its error in the pace's)
            val kp = pp / s
            pace = (pace + kp * nu).coerceIn(-PACE_MOST, PACE_MOST)
            val a = -kp * hg; val b = 1 - kp * hp
            val gp = a * gainVar + b * gpCov
            paceVar = a * a * gainVar + 2 * a * b * gpCov + b * b * paceVar + kp * kp * (s - hph)
            gpCov = gp
        }
        gainVar = maxOf(gainVar, 1e-9); paceVar = maxOf(paceVar, 1e-18)
        check()
    }

    // ai: the process noise up to camera time t: the gain's (GAIN_Q) and the pace's (PACE_Q), each a second
    private fun age(t: Long) {
        if (paceAt != 0L && t > paceAt) { val s = (t - paceAt) / 1e9; gainVar += GAIN_Q * s; paceVar = minOf(PACE_VAR0, paceVar + PACE_Q * s) }
        paceAt = maxOf(paceAt, t)
    }

    // ai: The newest N_EVID looks (of the newest LONG_NS) re-check the pace: each says the pace at the gain as it is, a vote each
    // ai: (a jump of the view's spoils a look whatever its own error says, so none outweighs the rest); where their
    // ai: median is beyond Z of its error (a median's: pi / 2 of a look's variance over their count, a look's the larger
    // ai: of what its readings and the gain say, their median, and what the looks' spread about the median says: the
    // ai: view's wander as it is, not as a hand's is supposed) and the pace's own, the pace is the median and its
    // ai: variance the median's.
    private fun check() {
        // ai: (the looks of the newest LONG_NS: an older one says what the pace was)
        while (evid.isNotEmpty() && evid.first()[7] < paceAt - LONG_NS) evid.removeFirst()
        if (evid.size < N_EVID) return
        val n = evid.size; val ps = DoubleArray(n); val vs = DoubleArray(n)
        for (i in 0 until n) { val e = evid[i]; ps[i] = (e[0] - e[1] * gain) / e[2]; vs[i] = (e[4] + e[1] * e[1] * gainVar) / (e[2] * e[2]) }
        val m = median(ps)
        val mad = median(DoubleArray(n) { abs(ps[it] - m) })
        val err2 = Math.PI / 2 * maxOf(median(vs), (1.4826 * mad) * (1.4826 * mad)) / n
        if (abs(m - pace) > Z * Math.sqrt(err2 + paceVar)) {
            if (DEBUG) say("phase: DEBUG the newest %d looks say the pace is %.0f us a second (to %.0f), not %.0f (to %.0f)".format(n, m * 1e6, Math.sqrt(err2) * 1e6, pace * 1e6, sure * 1e6))
            pace = m.coerceIn(-PACE_MOST, PACE_MOST); paceVar = minOf(PACE_VAR0, err2); gpCov = 0.0
            // ai: what those looks told the gain they told it at a pace that was wrong: the gain goes back to what it
            // ai: was before the oldest of them, its variance then and its process noise since
            val o = evid.first()
            gain = o[5]; gainVar = o[6] + GAIN_Q * maxOf(0.0, (paceAt - o[7]) / 1e9)
            evid.clear()
        }
    }

    private fun median(v: DoubleArray): Double { val s = v.sortedArray(); return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2 }

    // ai: The pilots' hold: what the newest frames hold of their neighbours, and the move it asks.
    // ai:   both neighbours in them (a top that bends: no phase is clean): later by the gain times a - b;
    // ai:   one neighbour only (the edge of a top that is flat, where a capture holds neither): out of it and a
    // ai:     step of CROSS in. Back at the same edge after standing clear of it, the two clocks' slide is the
    // ai:     step over the time it took, and the pace takes it (in the flat no reading shows a slide). At the
    // ai:     other edge straight from the step, the top is narrower than the step: its width is known, and the
    // ai:     hold goes to its middle;
    // ai:   neither: the hold stands; LONG_NS of it from the last step, and the width a step met may be the top's
    // ai:     no longer (a window that shortened widens it): the next step is CROSS again and measures it afresh.
    // ai: Every reading that shows a neighbour, with the one before, also tells the gain and the pace (`learn`),
    // ai: and the move allows for what the hold has slid since its frames were captured.
    private fun held() {
        val fs = window.filter { it.ts > settle }
        if (fs.size < L_NEED) return
        level = maxOf(level * 0.98, fs.sumOf { it.blocks }.toDouble() / fs.size)
        // ai: the newest second's frames where they are enough (what the hold holds now, not a mean over a long
        // ai: stand), else all since the last move
        val est = leaks(maxOf(settle, fs.last().ts - RECENT_NS)) ?: leaks(settle)
        if (est != null) lastLeak = est
        if (est == null) {
            // ai: no frame tells its neighbours apart. Deep in a mix (half the frames' r under DEEP: no capture is
            // ai: mostly one picture) nothing has a side: a quarter of a refresh later, and look again. Only where
            // ai: half the frames have no sure sign: frames that have one and are just too few yet (a small batch,
            // ai: its ends without a neighbour) will tell the side with the next batch, and a blind step is not it
            val mid = fs.map { abs(it.r) }.sorted()[fs.size / 2]
            val signed = fs.count { abs(it.r) >= R_FLASH && abs(it.r2) >= R_FLASH }
            if (mid < DEEP && 2 * signed < fs.size) {
                line += Math.round(refresh / 4); move = true; prevT = 0; edgeDir = 0; flatHalf = 0; stood = false
                if (DEBUG) say("phase: DEBUG deep in a mix (|r| %.2f): a quarter later".format(mid))
            }
            return
        }
        val e = est.a - est.b; val se = Math.hypot(est.sa, est.sb)
        // ai: a neighbour is in the frames where its share is Z of its errors and LEAK at least (under that it costs
        // ai: no block, and at a top that is nearly flat both shares hover about nothing)
        val sigA = est.a > maxOf(LEAK, Z * est.sa); val sigB = est.b > maxOf(LEAK, Z * est.sb)
        // ai: readings of frames that overlap say one thing twice: a reading counts once its frames are a
        // ai: RECENT on from the last one's (or a move has been made since, which starts the frames afresh)
        if (prevT != 0L && prevHop == 0.0 && est.t - prevT < RECENT_NS) return
        val was = pace
        // ai: a pair tells the gain and the pace where a - b went straight with the hold's place between them: both
        // ai: readings showing the same neighbours (across a flat top it stands still whatever the hold does)
        val side = (if (sigA) 1 else 0) or (if (sigB) 2 else 0)
        if (prevT != 0L && side != 0 && side == prevSide && est.t > prevT) learn(est.t, e - prevE, (est.t - prevT) / 1e9, prevHop + (est.x - prevX) / 1e6, prevPace * 1e3, 1 / (se * se + prevSe * prevSe))
        if (DEBUG) say("phase: DEBUG leaks %d frames: before %.3f (to %.3f), after %.3f (to %.3f), gain %.1f ms, pace %.0f%s".format(est.n, est.a, est.sa, est.b, est.sb, gain, pace * 1e6,
            if (flatHalf > 0) ", a flat top %.1f ms wide".format(2 * flatHalf / 1e6) else if (edgeDir != 0) ", %.1f ms in from the %s edge".format(edgeIn / 1e6, if (edgeDir > 0) "early" else "late") else ""))
        var hop = 0L
        if (sigA && sigB) {
            edgeDir = 0; flatHalf = 0
            if (abs(e) >= Z * se) hop = Math.round(gain * e * 1e6)
            else if (!balanced && paceVar < PACE_VAR0) {
                balanced = true
                say("phase: the hold stands: %.0f%% of the picture before and %.0f%% of the one after in a capture; the pace %.0f us a second (to %s), the line %.1f ms a unit".format(
                    100 * est.a, 100 * est.b, pace * 1e6, "%.0f".format(sure * 1e6), gain))
            }
        } else if (sigA || sigB) {
            val dir = if (sigA) 1 else -1
            var edge = Math.round(gain * (if (sigA) est.a else est.b) * 1e6)
            if (edgeDir == dir && edgeClean && est.t - edgeT >= PACED_NS) {
                // ai: back at the edge it stepped in from, having stood inside the top meanwhile: the slide is
                // ai: the step (and what it is past the edge now) over the time since the step was reckoned from
                // ai: as a look at (G, p): p dt - dir share G = dir edgeIn + pace dt (ms), the two edges' readings' errors
                val dt = (est.t - edgeT).toDouble(); val dtS = dt / 1e9
                observe(est.t, dir * edgeIn / 1e6 + pace * dtS * 1e3, -dir * (if (sigA) est.a else est.b), dtS * 1e3, (2 * gain * Math.max(est.sa, est.sb)).let { it * it }, dtS, false)
                say("phase: the hold slid %.1f ms in %.1f s: the pace %.0f us a second (to %.0f)".format((edgeIn + edge) / 1e6, dt / 1e9, pace * 1e6, sure * 1e6))
            } else if (edgeDir == -dir && est.t > edgeT && !edgeClean) {
                // ai: the other edge, straight from a step in from the first: the top is narrower than the step,
                // ai: as wide as what was stepped, and the hold goes to its middle. What the captures moved (their own
                // ai: places) is the top's width and both edges' depths, each the gain times its share; a gain that puts
                // ai: the depths alone past the move is too large (a width is never less than 0), and the move over the
                // ai: change of a - b is the most it can be (else every step from one edge overshoots to the other, and
                // ai: readings of one neighbour each never pair to tell the gain)
                val moved = abs(prevHop * 1e6 + (est.x - prevX)); val de = abs(e - prevE)
                if (prevT != 0L && prevSide == 3 - side && gain * de * 1e6 > moved) { gain = (moved / de / 1e6).coerceIn(GAIN_LEAST, GAIN_MOST); edge = Math.round(gain * (if (sigA) est.a else est.b) * 1e6) }
                flatHalf = maxOf(BAND_NS, (edgeIn - edge) / 2)
                say("phase: the top is %.1f ms wide".format(2 * flatHalf / 1e6))
            } else if (edgeDir == -dir) flatHalf = 0
            val step = if (flatHalf > 0) flatHalf else CROSS_NS
            hop = dir * (edge + step)
            // ai: the step is reckoned from now where the pace was just put right (the move below allows for the
            // ai: slide since the frames); where it was not, the hold slides on as it did, and the step is
            // ai: reckoned from the frames' own time
            edgeDir = dir; edgeIn = step; edgeT = if (pace != was) now else est.t; edgeClean = false
        } else {
            if (edgeDir != 0 && est.t > edgeT) edgeClean = true   // ai: (its frames are after the step landed: settle)
            if (flatHalf > 0 && est.t - edgeT >= LONG_NS) flatHalf = 0
        }
        // ai: the frames are old by the time they are read, and the hold slid on meanwhile at what the pace was
        // ai: off by: the move allows for it (the pace is right from now on)
        hop = hop.coerceIn(-Math.round(refresh / 2), Math.round(refresh / 2))
        if (hop == 0L) stood = true
        // ai: (the next pair's move is this one without that allowance: its line reckons the slide at the new pace
        // ai: from the frames' own time)
        if (side != 0) { prevE = e; prevSe = se; prevT = est.t; prevHop = hop / 1e6; prevPace = pace; prevSide = side; prevX = est.x } else prevT = 0
        if (hop != 0L || pace != was) hop += Math.round((pace - was) * (now - est.t))
        if (hop != 0L) { line += hop; move = true }
    }

    // ai: track's standing, for the log's line and the snapshot
    private val standing get() = if (line < 0) "no frames yet" else "holds"
    private val landMs get() = lands.maxOfOrNull { it[1] }?.let { it / 1e6 } ?: Double.NaN
    private fun snap() {
        val e = lastLeak
        state = State("pilots", standing, e?.a ?: Double.NaN, e?.b ?: Double.NaN, e?.k ?: Double.NaN,
            if (e == null) Double.NaN else Math.hypot(e.sa, e.sb), e?.n ?: 0, gain, pace * 1e6, stood, edgeDir, 2 * flatHalf / 1e6, delaysAll,
            if (base == 0L) Double.NaN else refresh / 1e6, multiple, frame / 1e6, landMs, row)
    }

    // ai: for the log, every few seconds: what is held or owed, the delays since the last call, the hold's newest
    // ai: leaks, and what a bad start is told by: the refresh on the camera's clock, the refreshes a capture spans and
    // ai: the camera's frame, the delays' longest landing, the code's row (the prefix stays as tools/phone's readers
    // ai: match it)
    fun line(tsNs: Long): String? {
        val d = delays; delays = 0
        if (mode == Mode.Off) return null
        return "phase: track%s, %s, the pace %.0f us a second (to %s), %.1f blocks a frame, %d delays%s; the refresh %s ms, %d a frame of %.3f ms, delays landing in %s, the code at row %.2f".format(
            if (line >= 0) " by the pilots" else "", standing, pace * 1e6, if (paceVar >= PACE_VAR0) "?" else "%.0f".format(sure * 1e6), level, d,
            lastLeak?.let { if (standing == "holds") ", %.0f%% of the picture before and %.0f%% of the one after in a capture".format(100 * it.a, 100 * it.b) else "" } ?: "",
            if (base == 0L) "?" else "%.4f".format(refresh / 1e6), multiple, frame / 1e6, if (lands.isEmpty()) "?" else "%.0f ms, %d captures after the mark".format(landMs, lands.maxOf { it[2] }), row)
    }

    companion object {
        val DEBUG = System.getProperty("phase.debug") == "1"   // ai: tools/phasesim: the hold's readings
        // ai: the most the pace may be (and how far a camera's frame may be off a whole number of refreshes); how
        // ai: late a capture may start of the line before a period less is asked; the span of frames, places and
        // ai: landings the hold keeps; how many of its errors a reading
        // ai: must be; the least time a flat top's edge reached again is reckoned over; the frames of camera time after
        // ai: which a delay nothing was heard of is given up (and the captures after its mark a taken delay may land
        // ai: while none has landed)
        const val PACE_MOST = 5e-3
        const val TOOTH_NS = 1_000_000L
        const val LONG_NS = 12_000_000_000L
        const val Z = 3.0
        const val PACED_NS = 1_000_000_000L
        const val LOST = 60
        // ai: the pilots: |r| at which a reading's sign is not noise; the least error a reading is given (its own
        // ai: at LIZARD-512 is about 0.02 over half the blocks); the frames an estimate of the leaks needs; the
        // ai: median |r| under which the frames are deep in a mix; the ms the line moves a unit of a - b before
        // ai: any move has measured it (under twice the least a display's own is, so a first move cannot grow the
        // ai: error: a window of 9 ms that changes pictures at one instant has 9, the S26's arithmetic about 20),
        // ai: and the least and most a measurement may say; the step in from the edge of a flat top; the span of
        // ai: the newest frames the leaks are read from (and the intervals the camera's frame is read from); the share
        // ai: of a neighbour under which a capture is taken to hold none of it; what a pair of readings keeps of its
        // ai: weight in the fit with each pair after it; how early a capture is delayed back onto the line; the code's
        // ai: row before a frame has told it (the frame's middle, where the receiver's crops are)
        const val R_FLASH = 0.25
        const val R_NOISE_MIN = 0.005
        const val L_NEED = 12
        const val DEEP = 0.35
        const val GAIN_MS = 12.0
        const val GAIN_LEAST = 2.0
        const val GAIN_MOST = 40.0
        const val CROSS_NS = 2_000_000L
        const val RECENT_NS = 1_000_000_000L
        const val LEAK = 0.02
        const val FADE = 0.85
        const val BAND_NS = 250_000L
        const val ROW = 0.5
        // ai: the estimate (`observe`). Before any look: the pace as far off as two clocks may be, the gain anywhere
        // ai: between its least and most (a uniform's variance). The view's wander, a random walk of the code's read
        // ai: time: a hand-held code wanders over half the frame's readout within a minute, re-aims and all (the S26's
        // ai: 6.26 ms readout: (3.13 ms)² a minute); a hand still aiming sweeps the code across
        // ai: half the frame in a second, 3 ms a second of read time, for the session's first three seconds (AIM_NS),
        // ai: three readings of RECENT_NS, which the newest N_EVID looks' median outvotes (seven outvote three). Process noise:
        // ai: the gain is half the window over the code, which a hand moves about a millisecond in ten seconds, so
        // ai: (0.5 ms a unit)² in ten seconds; the slide is the clocks' (which part by about 1 ppm a minute as a phone
        // ai: warms) and a hand's slow turn (about a degree a second, 0.15 ms of read time a second on the S26, which
        // ai: changes within about ten seconds): (0.15 ms a second)² in ten seconds.
        const val PACE_VAR0 = PACE_MOST * PACE_MOST
        const val GAIN_VAR0 = (GAIN_MOST - GAIN_LEAST) * (GAIN_MOST - GAIN_LEAST) / 12
        const val VIEW_MS2 = 3.13 * 3.13 / 60
        const val AIM_MS2 = 9.0
        const val N_EVID = 7
        const val AIM_NS = 3_000_000_000L
        const val GAIN_Q = 0.025
        const val PACE_Q = 0.15e-3 * 0.15e-3 / 10
    }
}
