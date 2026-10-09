package dev.lizard.receiver

import kotlin.math.abs

// ai: The camera's phase against the display (2026-09-30).
// ai: A capture that holds a change of the painted frame is two symbols and reads little, and whether captures hold
// ai: one depends on where they start in the display's refresh. A camera and a display each keep their own clock, so
// ai: that place slides: the S26's frame is 16.650 ms, 1.1 ms a second against a 60.00 Hz display, every phase in
// ai: 16 s. One frame made longer moves every frame after it (Engine.delay). That is the only lever: a delay of x
// ai: moves the phase x later, and a delay of a whole frame less x moves it x earlier (a capture skipped).
// ai:
// ai: Track (2026-09-30; auto, the arm before it, deleted 2026-10-01 after losing the A/B): a search that folds
// ai: its frames' timestamps by every period the display might have, a hold at the phase that reads the most
// ai: blocks, and a pace that is measured, never a share of the way to an estimate (its comment is above `search`).
// ai: The pilots (research/SPEC.md 7.3; since 2026-09-30, signed since 2026-10-01, after a night of a meter on |r|
// ai: alone, which is the same whichever neighbour a
// ai: capture leaks from: STATUS "The pilots' phase meter", "Signed pilots"). A sender that flashes paints its count
// ai: of pictures mod 4 into the block tails, bit 0 on the even blocks and bit 1 on the odd, and a capture reads r
// ai: (the even blocks) and r2 (the odd): their signs the count of the picture it mostly holds, and their sizes
// ai: what it holds of the pictures either side, each neighbour lowering the reading whose bit it does not share.
// ai: So where the frames flash (`flashing`) track needs no search and no sweep:
// ai:   the hold begins where the camera is (`here`), on the first frames;
// ai:   the leaks (`leaks`): over the frames since the last move, each reading is K less twice K times the share
// ai:     of each neighbour whose bit differs (which neighbours do is read off the neighbouring captures' counts,
// ai:     so a picture the screen showed twice or skipped is counted for what it was, not guessed): K, the share
// ai:     a of the picture before and b of the one after by least squares, each reading by its own error;
// ai:   the move: a capture that starts too early holds some of the picture before, one too late some of the
// ai:     next, so the line goes later by `gain` ms a unit of a - b wherever that is Z of its errors from 0; the
// ai:     gain is measured by every move (the move over what it changed a - b by). No frame's blocks are read
// ai:     for it, nothing is fitted against phase, and the camera is kept on the line, not dithered about it;
// ai:   deep in a mix (no capture mostly one picture, so no count): a quarter of a refresh later, and look again;
// ai:   the pace as below: each move of the line a TI-th of the way, and the hold's own spacing from the first
// ai:     balance on, to what the leaks leave of the line's place over the time held.
// ai: Frames that do not flash (an older sender, `usePilots` off for the A/B), or that carry no side (a format of one
// ai: block has bit 0 alone: r and no r2, so a capture says how much of the neighbours it holds and not which; the
// ai: decoders report the odd reading as NaN there), are read by their blocks as below.
// ai: Hold and Scan are the measuring tools that came first, on a grid of `gridNs` (a 60 Hz refresh on the camera's
// ai: clock): a hold of each capture's start at one point of it, and a scan of every point a step apart, a line each,
// ai: ending as a hold at the middle of the longest run of points that read as well as the best.
// ai: Every call is on the engine's thread.
class PhaseLock(private val say: (String) -> Unit, private val gridNs: Long = GRID_NS, private val bandNs: Long = BAND_NS) {
    enum class Mode { Off, Hold, Scan, Track }
    // ai: r, rSd: the pilots' reading over the even blocks and its standard error; r2, r2Sd: over the odd (NaN: none)
    class Frame(val ts: Long, val blocks: Int, val found: Boolean, val r: Double = Double.NaN, val rSd: Double = Double.NaN, val r2: Double = Double.NaN, val r2Sd: Double = Double.NaN)
    // ai: The lock's state for the stats row and a replay's meta (Engine.phaseState, MainActivity's poll; 2026-10-04): a
    // ai: snapshot made on the engine's thread at each decision, read from any. `arm`: pilots or blocks; `what`: the
    // ai: word line() uses (holds, placing, searching, resting, no frames yet); the newest estimate of the leaks (a
    // ai: and b the shares of the pictures before and after in a capture, k what a capture of one picture reads, se
    // ai: the error of a - b, n its frames; NaN with none); the gain (ms a unit of a - b) and the pace (us a second);
    // ai: whether the hold stands; the edge of a flat top it stepped in from (1 early, -1 late, 0 none) and the top's
    // ai: width (ms); the delays asked since the mode was set.
    class State(val arm: String, val what: String, val a: Double, val b: Double, val k: Double, val se: Double, val n: Int,
                val gain: Double, val pace: Double, val stood: Boolean, val edge: Int, val flatMs: Double, val delays: Int)
    @Volatile var state: State? = null
        private set

    private class Tally { var frames = 0; var short = 0; var blocks = 0L }

    var mode = Mode.Off
        private set
    private var waiting = false     // ai: a delay asked and not yet seen in a result
    private var delays = 0
    private var delaysAll = 0       // ai: delays asked since the mode was set (State)
    private var now = 0L            // ai: the newest capture's timestamp
    private var seen = 0L           // ai: the newest frame of the series already counted, ns
    private var plain = 0L          // ai: the camera's own frame: the last interval between two frames not delayed
    private var wasDelayed = false

    // ai: hold and scan
    private var base = -1L          // ai: the held point, ns on the grid; -1 none
    private val tally = HashMap<Long, Tally>()
    private var point = -1L         // ai: scanning: the point being counted; -1 not scanning
    private var start = -1L         // ai: the scan's first point

    // ai: track
    private var pace = 0.0          // ai: delay owed a unit of time (ns a ns); negative: earlier
    private var rests = 0           // ai: rests in a row: each twice as long as the one before, up to REST_MOST
    private var settle = 0L         // ai: frames after this one were captured after the last hop
    private var rest = 0L           // ai: no moves until then
    private var ref = 0L            // ai: the camera's own frame as it was when track first searched: what the pace is a share of
    private var out = 0             // ai: delays asked and not yet seen in a result, and the captures since the newest was asked
    private var outAge = 0
    private var searching = false
    private var searchT = 0L        // ai: a search: its first capture (its moves stop SEARCH_NS after it)
    private var ramp = 0.0          // ai: a search: ns of its moves not yet asked
    private val sought = ArrayList<Frame>()   // ai: a search's registered frames, in capture order, their blocks as read
    private var fails = 0           // ai: searches in a row that found nothing to hold
    private var line = -1L          // ai: the hold: the start of a capture on the phase it holds, the one nearest the newest capture; -1: none
    private var line0 = 0L          // ai: `line` when the hold began (0: not begun), and the refreshes from there to `line`
    private var slots = 0L
    private var move = false        // ai: the line was moved: the camera follows at once, not a tooth later
    private var moving = false      // ai: the delay that follows it is on its way
    private var bias = 0L           // ai: the camera is held this far later than the line (a quarter tooth either way: the dither)
    private var snap = false        // ai: the bias just changed: the camera follows at once
    private var askedT = 0L         // ai: when a delay was last asked
    private val window = ArrayDeque<Frame>()  // ai: the hold's registered frames, the newest LONG_NS, their blocks as read
    private var since = 0L          // ai: frames captured after this are the hold's; MAX: the camera is not on the line yet
    private var level = 0.0         // ai: blocks a frame where it reads (a search's, then the hold's newest frames)
    private var wide = 0L           // ai: the width of the phases that read, by the last search
    private var sure = Double.MAX_VALUE       // ai: how well the pace is known (ns a ns); MAX: not at all
    private var anchor = 0.0        // ai: the pace as last measured: the hold's moves push the pace about it, no further than `sure`
    private var proven = false      // ai: the pace is one a hold has measured, not only what a search told
    private var lastAt = 0L         // ai: the search before: a capture start on the middle of its arc, and the arc's width; 0: none
    private var lastWide = 0L
    private var flashing = false    // ai: the newest frames flash (the pilots): track holds by their leaks
    private var plainly = false     // ai: and they have been seen not to (an older sender): a search, and blocks
    var usePilots = true            // ai: off: the frames are read by their blocks whatever they flash (debug.lizard.pilots 0, the A/B)
    private val lately = ArrayDeque<Frame>()   // ai: the newest registered frames with a reading, for `flashing`
    // ai: the pilots' hold: whether the line is theirs (begun by `here`); ms the line moves a unit of a - b. The
    // ai: reading before (a - b, its error, its frames' mean time; 0: none that showed a neighbour), the move made
    // ai: on it (ms) and the pace it was made at; and the sums of the fit of the gain and the display's pace to the
    // ai: readings' history (`learn`).
    private var byLeaks = false
    private var gain = GAIN_MS
    private var prevE = 0.0
    private var prevSe = 0.0
    private var prevT = 0L
    private var prevHop = 0.0
    private var prevPace = 0.0
    private var prevSide = 0        // ai: which neighbours it showed: 1 the one before, 2 the one after, 3 both
    private val fitS = DoubleArray(5)
    private var pairs = 0           // ai: the pairs of readings in that fit
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
    // ai: Whether the lock wants its frames soon: track, with a symbol registered in the last RECENT (`foundT`:
    // ai: the newest registered frame fed), until a hold by the pilots first asks no move (and again from a mix too
    // ai: deep to read). The receiver decoded in small batches while it was on until 2026-10-08, when the native
    // ai: receiver stopped waiting for a batch (a launch as soon as a frame is staged and a lane is free); only
    // ai: tools/phasesim reads it now, modelling a batching receiver.
    private var stood = false
    private var foundT = 0L
    val soon get() = mode == Mode.Track && usePilots && !plainly && !stood && now >= rest && foundT != 0L && now - foundT < RECENT_NS
    private var looked = 0L         // ai: when the blocks' search or hold last looked at its frames
    fun set(m: Mode, holdUs: Long = 0) {
        if (m == mode && (m != Mode.Hold || holdUs * 1000 % gridNs == base)) return
        mode = m
        tally.clear(); point = -1; start = -1
        base = if (m == Mode.Hold) holdUs * 1000 % gridNs else -1
        pace = 0.0; rests = 0; settle = now; rest = 0
        ref = 0; out = 0; searching = false; sought.clear(); fails = 0; line = -1; line0 = 0; move = false; moving = false
        window.clear(); since = 0; level = 0.0; wide = 0; sure = Double.MAX_VALUE; anchor = 0.0; proven = false; lastAt = 0
        flashing = false; plainly = false; lately.clear()
        byLeaks = false; gain = GAIN_MS; prevT = 0; fitS.fill(0.0); pairs = 0; edgeDir = 0; flatHalf = 0; stood = false; looked = 0; foundT = 0
        lastLeak = null; state = null; delaysAll = 0
        say(when (m) {
            Mode.Off -> "phase: off"
            Mode.Hold -> "phase: held at %.3f ms of %.3f".format(base / 1e6, gridNs / 1e6)
            Mode.Scan -> "phase: scan"
            Mode.Track -> "phase: track"
        })
    }

    // ai: The camera was opened again (a setting changed, the app came back): its frames start anywhere in the
    // ai: display's refresh, but the two clocks are the ones they were, so the pace stands and only the place is
    // ai: looked for again.
    fun reopened() {
        waiting = false; now = 0; plain = 0; wasDelayed = false
        settle = 0; rest = 0
        out = 0; searching = false; sought.clear(); fails = 0; line = -1; line0 = 0; move = false; moving = false; window.clear(); since = 0; level = 0.0
        flashing = false; plainly = false; lately.clear()
        byLeaks = false; prevT = 0; edgeDir = 0; stood = false; looked = 0; foundT = 0
        lastLeak = null; state = null
        if (mode == Mode.Scan) { tally.clear(); point = -1; start = -1; base = -1 }
    }

    private fun wrap(d: Long): Long { var e = d % gridNs; if (e > gridNs / 2) e -= gridNs; if (e < -gridNs / 2) e += gridNs; return e }
    private val tracking get() = mode == Mode.Track
    // ai: track: the display's refresh on the camera's clock, as the pace has it; and the step it lets the camera
    // ai: slide before a delay puts it back: a quarter of the phases that read, by the last search, at most TOOTH
    private val period get() = ref * (1 + pace)
    private val tooth get() = if (wide > 0) (wide / 4).coerceIn(bandNs, TOOTH_NS) else TOOTH_NS

    // ai: A capture's result: its timestamp, and whether it was a delayed frame. Returns the microseconds to delay
    // ai: the next frame by, 0 for none.
    fun onCapture(tsNs: Long, delayed: Boolean): Long {
        if (tsNs == 0L) return 0
        // ai: a camera's first capture: the frames the receiver still holds from before it say where another was
        if (now == 0L) { seen = maxOf(seen, tsNs - 1); settle = maxOf(settle, tsNs - 1) }
        val dt = if (now != 0L) tsNs - now else 0L
        if (dt > 0 && !delayed && !wasDelayed) plain = dt
        wasDelayed = delayed
        now = tsNs
        // ai: the delay asked has landed
        if (delayed) waiting = false
        if (mode == Mode.Off) return 0
        if (tracking) return track(tsNs, dt, delayed)
        var us = 0L
        // ai: a scan starts where the camera is, so its first point costs no jump
        if (base < 0 && point < 0) { point = tsNs % gridNs; start = point }
        if (!waiting) {
            // ai: early by more than the band: delayed by what it is early. Late (a camera slower than the grid,
            // ai: which no delay can pull back): delayed the rest of a period, a capture skipped.
            val err = wrap(tsNs % gridNs - (if (point >= 0) point else base))
            us = if (err < -bandNs) -err / 1000 else if (err > 2 * bandNs) (gridNs - err) / 1000 else 0L
        }
        if (us > 0) { waiting = true; delays++; delaysAll++ }
        return us
    }

    // ai: The receiver's newest frames (ns on the camera's clock, verified blocks, whether a symbol was registered) and
    // ai: the blocks a frame the word names.
    fun onWindow(given: List<Frame>, version: Int) {
        val frames = captures(given)
        val newest = frames.maxOfOrNull { it.ts } ?: 0L
        // ai: before a camera's first capture the receiver's frames are another session's: counted as seen, no more
        if (now == 0L) { seen = maxOf(seen, newest); return }
        if (tracking && version > 0) tracked(frames.filter { it.ts > seen && it.found }.sortedBy { it.ts }, version)
        else if (mode == Mode.Scan && version > 0 && point >= 0) scan(frames, version)
        seen = maxOf(seen, newest)
    }

    // ai: One frame a capture. The 2:1 crop gives the receiver two frames a capture (its halves, the capture's
    // ai: timestamp both), and they are one look at one mix of pictures: as two frames, each half's neighbouring
    // ai: capture in `leaks` was its own twin (a left half read its after-leak as none, a right half its before-leak)
    // ai: and each capture's mix counted twice as evidence (every reading's error about 1 / sqrt 2 of its own, so the
    // ai: hold's tests passed on noise and `learn` trusted its first pairs: on the S26 2026-10-03, in 2:1, the pace
    // ai: went from 1,052 to its bound, 5,000 us a second, "to 40", within 4 s of a hold beginning, and every hold
    // ai: after it walked off the picture and was pulled back each second; STATUS "The oscillation in 2:1"). Frames
    // ai: of one timestamp are one: each pilot reading their mean by inverse variance, the blocks their mean (a
    // ai: symbol's, as the word's version is), found where either was.
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
        return Frame(g[0].ts, Math.round(g.sumOf { it.blocks }.toDouble() / g.size).toInt(), g.any { it.found }, r, rSd, r2, r2Sd)
    }

    // ai: Track. It holds the phase at which a capture reads the most blocks. Everything it knows of where a capture
    // ai: sat comes from the capture's own timestamp, never from what was asked of the camera, so a delay that lands
    // ai: late, short or not at all costs nothing but time.
    // ai:   a search, at the start and whenever the hold stops reading: the camera's phase ramps later at
    // ai:     SEARCH_RATE (a delay asked as the last lands; on the S26 a sixteenth of a frame every seven captures,
    // ai:     9 ms a second, faster than any drift). Its frames are folded by every period the display might have
    // ai:     (`Fold`): at the right one the frames that read fall in one arc of the refresh. One pass of the arc
    // ai:     fits any period; when an edge of the arc has been passed twice the period is told, to how finely the
    // ai:     frames sample that edge over the time between the two passes, and the search ends (3 to 4 s; a pace
    // ai:     known from before needs one pass). Nothing has to fail first and no width is assumed.
    // ai:   the hold: a line of capture starts, a period apart, on the arc's middle. A capture that has slid a
    // ai:     `tooth` early of it (the two clocks) is delayed back onto it, so the frames sit across a tooth of
    // ai:     phases, and a straight line through their blocks against phase and time (`fit`) has a slope where a
    // ai:     better phase lies to one side: the line moves a tooth that way for a slope in the newest frames, half
    // ai:     for one only in the longer span. The criterion is the blocks a frame, not the middle of the phases
    // ai:     that are not short.
    // ai:     Where the two clocks hardly slide, the camera is held a quarter tooth either side of the line in
    // ai:     turn, as often as the pace's error could carry the hold half a tooth unseen (the dither).
    // ai:   the pace: what a search told, then the hold's own spacing: the line began on the arc's middle and is
    // ai:     still on the arc, so its mean period since is the display's to half the arc over the time held, and
    // ai:     that is taken whenever it is the more precise; and two searches' arcs, a whole number of refreshes
    // ai:     apart. Between those, every move of the line also moves the pace its way (a line that has to keep
    // ai:     moving one way is a pace that is off), a TI-th of the move a second, and never further from the last
    // ai:     measurement than its error. A pace only a search told, whose hold stops reading before any hold has
    // ai:     measured it, is dropped, and the next search tells it afresh. Nothing takes a share of the way to an
    // ai:     estimate, and a hand's shake (no slope, under FELL frames) moves nothing.
    private fun search() {
        if (plain == 0L) return
        if (ref == 0L) ref = plain
        searching = true; searchT = now; ramp = 0.0; sought.clear()
        line = -1; line0 = 0; move = false; moving = false; window.clear(); since = Long.MAX_VALUE; level = 0.0
    }

    // ai: A capture (track): the delay to ask, us.
    private fun track(ts: Long, dt: Long, delayed: Boolean): Long {
        if (delayed && out > 0) { out--; if (out == 0 && moving) { moving = false; settle = ts } }
        // ai: a delay the camera never reported is given up, or nothing would be asked again
        if (out > 0 && ++outAge > LOST) { out = 0; moving = false }
        if (ref == 0L || ts < rest) return 0
        var ns = 0L
        if (searching) {
            if (ts - searchT < SEARCH_NS) {
                ramp += SEARCH_RATE * dt
                // ai: a step at most (the engine's thread may have stalled and let more owe: the frames must still
                // ai: sample every phase, so the rest waits)
                if (out == 0 && ramp >= ref / 64) { ns = minOf(ramp.toLong(), (ref / SEARCH_STEPS).toLong()); ramp -= ns }
            }
        } else if (line >= 0) {
            // ai: the line's point nearest this capture, and how early of it the capture starts
            val k = Math.round((ts - line) / period)
            if (k != 0L) { line += Math.round(k * period); slots += k }
            // ai: nothing is asked on a delayed frame's own timestamp: on the S26 it is already the later one, but a
            // ai: camera that stamps the delayed frame before its delay would be asked for the same delay twice
            if (out == 0 && !delayed) {
                val placing = since == Long.MAX_VALUE
                if (placing && abs(line - ts) < tooth / 2) { since = ts; settle = ts; line0 = line; slots = 0; bias = 0; askedT = ts }
                else {
                    // ai: the two clocks slide the camera across a tooth of phases, which is what a slope is read
                    // ai: from. Where they hardly slide, the camera is held a quarter tooth later than the line, then
                    // ai: a quarter earlier, in turn (a capture skipped every other turn), as often as the pace's
                    // ai: error could carry the hold half a tooth unseen: every second or two while the pace is
                    // ai: only what a search told, hardly ever once a long hold has measured it
                    val idle = if (sure == Double.MAX_VALUE) DITHER_NS else maxOf(DITHER_NS, (tooth / 2 / sure).toLong())
                    if (byLeaks) bias = 0
                    else if (!placing && !move && ts - askedT > idle) { bias = if (bias > 0) -tooth / 4 else tooth / 4; snap = true; askedT = ts }
                    val early = line + (if (placing) 0 else bias) - ts
                    val far = if (placing || move || snap) ref / 256 else tooth
                    // ai: early: delayed onto the line (the pilots' hold: as soon as it is a band early, a delay
                    // ai: costing nothing and its readings wanting no spread). Late (a camera slower than the
                    // ai: display, a line moved earlier): a frame less what it is late, a capture skipped
                    if (early >= (if (byLeaks) minOf(far, bandNs) else far)) ns = early else if (early <= -far) ns = ref + early
                    if (move) { if (ns != 0L) moving = true else settle = ts; move = false }
                    snap = false
                }
            }
        }
        if (ns <= 0) return 0
        out++; outAge = 0; delays++; delaysAll++; askedT = ts
        return maxOf(1, ns / 1000)
    }

    // ai: Whether the newest frames flash: of the consecutive pairs a refresh apart (the series leaves some captures
    // ai: out) that both read |r| >= R_FLASH (a sign that is not noise), at least FLASH_PAIRS, and a quarter of them
    // ai: of opposite signs (a picture the sender shows for k refreshes flips one pair in k; a sender that does not
    // ai: flash flips none).
    // ai: Null while too few pairs are readable (deep in a mix nothing has a sign): what was known stands.
    private fun flashNow(): Boolean? {
        val t = if (ref > 0) ref else plain
        if (t == 0L) return null
        var pairs = 0; var flips = 0
        var a: Frame? = null
        for (b in lately) {
            if (a != null && b.ts - a.ts < t * 3 / 2 && abs(a.r) >= R_FLASH && abs(b.r) >= R_FLASH) { pairs++; if ((a.r > 0) != (b.r > 0)) flips++ }
            a = b
        }
        if (pairs < FLASH_PAIRS) return null
        return flips >= maxOf(2, pairs / 4)
    }
    // ai: whether a frame reads (the blocks' hold): HELD of the level's blocks
    private fun reads(f: Frame): Boolean = f.blocks >= level * HELD

    // ai: The pilots' hold begun where the camera is: the line through the newest capture, the pace as it was known.
    private fun here() {
        if (plain == 0L) return
        if (ref == 0L) ref = plain
        searching = false; sought.clear()
        byLeaks = true; prevT = 0; wide = 0; edgeDir = 0; flatHalf = 0; balanced = false; stood = false; lastLeak = null
        line = now; line0 = 0; slots = 0; move = false; moving = false; window.clear(); since = Long.MAX_VALUE; level = 0.0
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
    // ai: r, bit 1 from r2), and so are the captures' before and after it, a refresh either side (a frame whose
    // ai: neighbour is not in the series, or any of the three with a reading under R_FLASH, whose sign may be
    // ai: noise, is left out). Each of its readings is then, in size, K - 2 K a [the bit differs from the count
    // ai: before] - 2 K b [it differs from the count after]: K what a capture of one picture reads, a and b the
    // ai: shares of the pictures before and after. Least squares over both readings of every frame, each by its
    // ai: own error, for K, K a and K b. Null: too few frames, or frames that do not tell the three apart (a
    // ai: sender that does not flash: no bit ever differs).
    private class Leak(val a: Double, val b: Double, val sa: Double, val sb: Double, val n: Int, val t: Long, val k: Double)   // ai: t: the frames' mean time; k: what a capture of one picture reads
    private fun leaks(from: Long, to: Long = Long.MAX_VALUE): Leak? {
        val fs = window.filter { !it.r.isNaN() && !it.r2.isNaN() }
        val t = period
        val m = Array(3) { DoubleArray(7) }
        for (k in 0 until 3) m[k][4 + k] = 1.0
        var n = 0; var ts = 0.0
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
        }
        if (n < L_NEED || !solve3(m)) return null
        val k = m[0][3] / m[0][0]
        if (k < R_FLASH) return null
        return Leak(m[1][3] / m[1][1] / k, m[2][3] / m[2][2] / k, Math.sqrt(maxOf(0.0, m[1][5] / m[1][1])) / k, Math.sqrt(maxOf(0.0, m[2][6] / m[2][2])) / k, n, from + Math.round(ts / n), k)
    }

    // ai: The receiver's new registered frames (track), in capture order, their blocks as read.
    private fun tracked(got: List<Frame>, version: Int) {
        if (got.isNotEmpty()) foundT = got.last().ts
        if (now < rest) return
        for (f in got) if (!f.r.isNaN()) lately.addLast(f)
        while (lately.size > FLASH_FRAMES) lately.removeFirst()
        val was = flashing
        val seen = flashNow()
        // ai: a side: a frame with both readings (a format of one block has r alone, and a hold needs which neighbour,
        // ai: not how much). The newest FLASH_FRAMES decide, so a format changed under the lock moves it within a
        // ai: second either way
        val sided = lately.count { !it.r2.isNaN() }
        flashing = usePilots && sided > 0 && (seen ?: flashing)
        // ai: frames with no reading at all (a decoder that reads none), or none with a side, are plainly not a
        // ai: flashing sender's for the hold: read by their blocks
        plainly = !usePilots || (if (seen != null && sided > 0) !seen else plainly || (got.isNotEmpty() && sided == 0))
        if (flashing != was) say(if (flashing) "phase: the frames flash: held by the pilots' leaks" else if (sided == 0 && lately.isNotEmpty()) "phase: the frames carry no side (one reading): read by their blocks" else "phase: the frames do not flash: read by their blocks")
        // ai: a search or a hold under way is the other kind's: frames that flash take the hold where it is, and
        // ai: frames seen not to flash give the pilots' hold up for a search
        if (flashing && (searching || (line >= 0 && !byLeaks))) { if (searching) here() else { byLeaks = true; prevT = 0; edgeDir = 0; flatHalf = 0 } }
        else if (plainly && byLeaks) { byLeaks = false; search() }
        val fs = got
        // ai: the pilots' hold at once, unless the frames are known not to flash: a sender flashes, and waiting to
        // ai: see it costs the start a second
        if (!searching && line < 0) { if (fs.isNotEmpty()) { if (plainly) search() else here() }; return }
        // ai: The frames come as the receiver reads them (a batch, or one at a time), not twice a second as they
        // ai: did to 2026-10-01. The pilots' hold reads them as they come. The blocks' search and hold look every
        // ai: LOOK, as the feed they were built and judged on gave them: the fold costs the engine's thread
        // ai: milliseconds, and a slope tried at every frame is tried thirty times as often at the same Z.
        val look = now - looked >= LOOK_NS
        if (searching) {
            for (f in fs) if (f.ts >= searchT) sought.add(f)
            if (look) { looked = now; searched(version) }
            return
        }
        for (f in fs) if (f.ts > since) window.addLast(f)
        while (window.isNotEmpty() && window.first().ts < window.last().ts - LONG_NS) window.removeFirst()
        if (byLeaks) { if (fs.isNotEmpty()) decide() } else if (look) { looked = now; decide() }
    }

    // ai: What the search's frames say so far. It ends the search when they tell the period (or, the pace known
    // ai: already, show a whole pass of the arc), and gives it up when its moves are over and they do not.
    private fun searched(version: Int) {
        val n = sought.size
        val over = now - searchT >= SEARCH_NS + SEARCH_LATE
        fun gaveUp(why: String, nothing: Boolean) {
            searching = false
            say("phase: a search $why")
            // ai: nothing to hold twice in a row (a display the camera's frames do not line up with, pixels too slow
            // ai: for any phase to read): no moves for REST, twice as long each time it happens again
            if (nothing && ++fails >= 2) {
                val long = minOf(REST_MOST, REST_NS shl minOf(rests, 8))
                say("phase: two searches and nothing to hold: resting %d s".format(long / 1_000_000_000))
                fails = 0; rests++; rest = now + long
            }
        }
        if (n < NEED) { if (over) gaveUp("whose frames did not come", false); return }
        val fold = Fold(LongArray(n) { sought[it].ts }, IntArray(n) { sought[it].blocks }, ref, version)
        if (fold.reads == 0) { if (over) gaveUp("in which no phase read", true); return }
        val told = fold.scan()
        if (told.every) {
            // ai: no phase is worse than another (a picture painted for several refreshes), once the moves have
            // ai: covered a refresh and a quarter: the hold is where the camera is
            if ((sought[n - 1].ts - searchT) * SEARCH_RATE < 1.25 * ref && !over) return
            searching = false; fails = 0; rests = 0
            level = fold.level; wide = ref; line = now; move = false
            say("phase: a search: every phase reads, %.1f blocks a frame".format(level))
            return
        }
        // ai: taken when it is the more precise, or when the pace lies outside its error (a search comes after a hold
        // ai: that did not last: fresh frames that put the period elsewhere are believed over a pace that failed)
        if (told.told && (told.err < sure || abs(told.pace - pace) > told.err)) { pace = told.pace; sure = told.err; anchor = pace; proven = false }
        if (sure == Double.MAX_VALUE) { if (over) gaveUp("that did not tell the display's period (%d of %d frames read)".format(fold.reads, n), true); return }
        var arc = fold.arc(pace)
        if (arc == null) { if (over) gaveUp("that found no arc at the pace", true); return }
        if (!told.told && arc.travel < period * 1.25 && !over) return
        // ai: two searches have each found the arc's middle, a whole number of refreshes apart: which number is
        // ai: plain when the period is known to a quarter refresh over the time between (by this search's own
        // ai: telling, or by the pace), and then that time over that number is the period, to the arcs' quarter
        // ai: widths over the time. It is what puts right a pace a search told wrongly and the hold fell off.
        val gap = (arc.at - lastAt).toDouble()
        if (lastAt != 0L && gap >= PACED_NS) {
            val by = if (told.told && told.err * gap < ref / 4) told.pace else if (sure * gap < ref / 4) pace else Double.NaN
            val err = (arc.wide + lastWide) / 4.0 / gap
            if (!by.isNaN() && err < sure) {
                pace = gap / Math.round(gap / (ref * (1 + by))) / ref - 1; sure = err; anchor = pace; proven = true
                arc = fold.arc(pace) ?: arc
            }
        }
        searching = false; fails = 0; rests = 0
        level = arc.level; wide = arc.wide; line = arc.at; move = false
        lastAt = arc.at; lastWide = arc.wide
        say("phase: a search: %d frames in %.1f s, %d read; %.1f ms of phases, %.1f blocks a frame; the pace %.0f us a second (to %.0f)".format(
            n, (sought[n - 1].ts - sought[0].ts) / 1e9, fold.reads, wide / 1e6, level, pace * 1e6, sure * 1e6))
    }

    // ai: A search's frames folded by a period of the display: the frames in the order of their phases, each read
    // ai: or not (`version`: the word's blocks a frame), and the arc of phases most likely to be the ones
    // ai: that read: on the arc a frame reads with probability p1, off it with p0 (a picture the sender shows for
    // ai: two refreshes reads at any phase; on the S26 and this desk's monitor a quarter of the captures off the
    // ai: arc read), both taken from the frames themselves. Apart from the lock so that it runs on recorded frames
    // ai: (tools/phasesim).
    class Told(val pace: Double, val err: Double, val told: Boolean, val every: Boolean)
    class Arc(val at: Long, val wide: Long, val travel: Double, val level: Double)
    class Fold(private val ts: LongArray, private val blocks: IntArray, private val ref: Long, version: Int) {
        private val n = ts.size
        private val read: BooleanArray
        val reads: Int
        val level: Double
        private val ph = DoubleArray(n)
        private val order = IntArray(n) { it }
        private var from = 0
        private var len = 0
        private var w1 = Math.log(0.95 / 0.25)
        private var w0 = Math.log(0.05 / 0.75)

        init {
            // ai: read or not: the frames' blocks fall in two heaps (a capture that holds a change reads next to
            // ai: nothing), and the cut between them is the one that leaves the two furthest apart for their sizes
            // ai: (Otsu's). The heap that reads must read an eighth of the word's blocks, or no frame reads.
            val hist = IntArray(blocks.max() + 2)
            for (b in blocks) hist[b]++
            var all = 0L; for (b in blocks) all += b
            var cut = 0; var most = -1.0; var below = 0; var sum = 0L
            for (c in 0 until hist.size - 1) {
                below += hist[c]; sum += c.toLong() * hist[c]
                if (below == 0 || below == n) continue
                val gap = sum.toDouble() / below - (all - sum).toDouble() / (n - below)
                val between = below.toDouble() * (n - below) * gap * gap
                if (between > most) { most = between; cut = c }
            }
            var up = 0; var upSum = 0L
            for (i in 0 until n) if (blocks[i] > cut) { up++; upSum += blocks[i] }
            // ai: one heap (every frame near the same blocks): all read, or none
            val one = most < 0 || (upSum.toDouble() / maxOf(1, up) - (all - upSum).toDouble() / maxOf(1, n - up)) * 8 < version
            val some = 8 * (if (one) all / n else upSum / maxOf(1, up)) >= version
            read = BooleanArray(n) { some && (one || blocks[it] > cut) }
            reads = read.count { it }
            level = if (reads == 0) 0.0 else (0 until n).filter { read[it] }.sumOf { blocks[it] }.toDouble() / reads
        }

        // ai: the frames' phases at the period ref (1 + d), in order (the order of the period before, put right: a
        // ai: small change of the period moves few frames past each other), and the run of them on the circle worth
        // ai: the most (ln of p1 over p0 for a frame that reads, of their complements for one that does not); its
        // ai: worth is returned, its first frame and length left in `from` and `len`
        private fun best(d: Double): Double {
            val t = ref * (1 + d)
            for (i in 0 until n) ph[i] = (ts[i] - ts[0]).toDouble() % t
            for (i in 1 until n) { val o = order[i]; var j = i - 1; while (j >= 0 && ph[order[j]] > ph[o]) { order[j + 1] = order[j]; j-- }; order[j + 1] = o }
            // ai: the best run that does not wrap, and the worst: the best that wraps is everything but the worst
            var top = -Double.MAX_VALUE; var ts0 = 0; var te = 0; var cur = 0.0; var cs = 0
            var low = Double.MAX_VALUE; var ls = 0; var le = 0; var cul = 0.0; var cls = 0
            var all = 0.0
            for (i in 0 until n) {
                val w = if (read[order[i]]) w1 else w0
                all += w
                if (cur <= 0) { cur = w; cs = i } else cur += w
                if (cur > top) { top = cur; ts0 = cs; te = i }
                if (cul >= 0) { cul = w; cls = i } else cul += w
                if (cul < low) { low = cul; ls = cls; le = i }
            }
            if (all - low > top && le - ls + 1 < n) { from = (le + 1) % n; len = n - (le - ls + 1); return all - low }
            from = ts0; len = te - ts0 + 1
            return top
        }

        private fun at(i: Int) = ph[order[((i % n) + n) % n]]
        private fun gap(x: Double, y: Double, t: Double) = ((y - x) % t + t) % t

        // ai: The period. Every drift is tried; the drifts whose arcs are as likely as the best to within NATS are
        // ai: the candidates, and the middle of them all the drift. It counts as told only when the frames pass one edge of
        // ai: the arc twice (one pass fits any period, and so does a search of just one refresh, the arc split
        // ai: across its ends): then its error is the larger of half the candidates' span and how far apart the
        // ai: frames either side of that edge sit, over the time between the two passes.
        fun scan(): Told {
            if (reads == n) return Told(0.0, 0.0, false, true)
            // ai: every drift, coarsely (the drifts an arc fits are a few tenths of a ms a second wide), then, with p1
            // ai: and p0 from the arc found, the drifts about the best finely. Coarse then fine is what the engine's
            // ai: thread can afford between captures: a scan over every fine drift stalled it for frames on the S26,
            // ai: and the search's steps grew with the stall
            val coarse = (PACE_MOST / DRIFT_COARSE).toInt()
            var lo = 0.0; var hi = 0.0
            run {
                val like = DoubleArray(2 * coarse + 1) { best((it - coarse) * DRIFT_COARSE) }
                val near = like.max() - NATS
                lo = (like.indexOfFirst { it >= near } - coarse) * DRIFT_COARSE; hi = (like.indexOfLast { it >= near } - coarse) * DRIFT_COARSE
                best((lo + hi) / 2)
                // ai: p1 and p0 from the arc just found, kept off 0 and 1
                var inR = 0; var outR = reads
                for (j in 0 until len) if (read[order[(from + j) % n]]) inR++
                outR -= inR
                val p1 = (inR.toDouble() / len).coerceIn(0.7, 0.995)
                val p0 = (if (n > len) outR.toDouble() / (n - len) else 0.25).coerceIn(0.005, 0.6)
                w1 = Math.log(p1 / p0); w0 = Math.log((1 - p1) / (1 - p0))
            }
            run {
                val from0 = maxOf(-PACE_MOST, lo - DRIFT_COARSE); val to0 = minOf(PACE_MOST, hi + DRIFT_COARSE)
                val m = ((to0 - from0) / DRIFT_STEP).toInt() + 1
                val like = DoubleArray(m) { best(from0 + it * DRIFT_STEP) }
                val near = like.max() - NATS
                lo = from0 + like.indexOfFirst { it >= near } * DRIFT_STEP; hi = from0 + like.indexOfLast { it >= near } * DRIFT_STEP
            }
            val d = (lo + hi) / 2
            val t = ref * (1 + d)
            best(d)
            // ai: an arc worth holding: a share of the frames, not nearly all or nearly none
            if (len < n / 10 || len > n - n / 10) return Told(d, 0.0, false, len > n - n / 10)
            val first = at(from); val last = at(from + len - 1)
            val edges = doubleArrayOf(first - gap(at(from - 1), first, t) / 2, last + gap(last, at(from + len), t) / 2)
            // ai: how far the phase had travelled by each frame (each interval's advance, to the nearest whole period)
            val u = DoubleArray(n)
            u[0] = ph[0]
            for (i in 1 until n) { var a = (ts[i] - ts[i - 1]).toDouble() % t; if (a > t / 2) a -= t; u[i] = u[i - 1] + a }
            val least = u.min() + ref / 12; val most = u.max() - ref / 12
            var err = Double.MAX_VALUE
            for (e in edges) {
                // ai: the passes of this edge with frames a step clear of it on both sides: the first and the last
                var t1 = 0L; var g1 = 0.0; var t2 = 0L; var g2 = 0.0
                var x = e + Math.ceil((least - e) / t) * t
                while (x <= most) {
                    var j = 1
                    while (j < n && u[j] < x) j++
                    if (j < n) { if (t1 == 0L) { t1 = ts[j]; g1 = u[j] - u[j - 1] } else { t2 = ts[j]; g2 = u[j] - u[j - 1] } }
                    x += t
                }
                if (t2 > t1) err = minOf(err, (abs(g1) + abs(g2)) / 2 / (t2 - t1))
            }
            if (err == Double.MAX_VALUE) return Told(d, 0.0, false, false)
            err = maxOf(err, (hi - lo) / 2)
            return Told(d, err, err <= DRIFT_ERR, false)
        }

        // ai: The arc at the drift d: a capture start on its middle (`at`), its width (from halfway between its first frame
        // ai: and the one before to halfway between its last and the one after), how far the frames' phase travelled,
        // ai: and the blocks of a frame that reads on it.
        fun arc(d: Double): Arc? {
            val t = ref * (1 + d)
            best(d)
            if (len < 3 || len >= n) return null
            val first = at(from); val last = at(from + len - 1)
            val e0 = first - gap(at(from - 1), first, t) / 2
            val wide = gap(e0, last + gap(last, at(from + len), t) / 2, t)
            var lo = ph[0]; var hi = ph[0]; var u = ph[0]
            for (i in 1 until n) { var a = (ts[i] - ts[i - 1]).toDouble() % t; if (a > t / 2) a -= t; u += a; if (u < lo) lo = u; if (u > hi) hi = u }
            var sum = 0L; var cnt = 0; var mid = 0.0
            for (j in 0 until len) { val k = order[(from + j) % n]; mid += (ts[k] - ts[0]).toDouble() / len; if (read[k]) { sum += blocks[k]; cnt++ } }
            // ai: the capture start on the arc's middle nearest the time the arc's own frames were taken: a period a
            // ai: little off then moves it least
            val at = e0 + wide / 2
            return Arc(ts[0] + Math.round(at + Math.round((mid - at) / t) * t), Math.round(wide), hi - lo, if (cnt > 0) sum.toDouble() / cnt else 0.0)
        }
    }

    // ai: A line through the hold's frames since `from`, and since the line last moved: blocks = a + g x + h t, x how
    // ai: late the frame started against the hold's line (ms) and t its time (s). g: blocks a ms later; h: blocks a
    // ai: second at one phase (the picture, the focus, a hand tiring: on the S26 it is often many of its errors with
    // ai: every frame reading), fitted so that g is not read from it. `spread`: the frames sit at phases apart
    // ai: enough, and not in step with time, to tell g from h. Only frames since the last move: what moved the line
    // ai: is spent, and frames from before it would move it again (an edge left behind reads as a slope ahead; on
    // ai: the S26 the first build's hold walked 3 ms across the phases that read that way, five moves in 10 s).
    private class Fit(val g: Double, val zg: Double, val spread: Boolean, val mean: Double)
    private fun fit(from: Long): Fit? {
        val fs = window.filter { it.ts > from && it.ts > settle }
        val n = fs.size
        if (n < NEED) return null
        val t0 = fs.last().ts
        val p = period
        val x = DoubleArray(n) { val v = (fs[it].ts - line).toDouble(); (v - Math.round(v / p) * p) / 1e6 }
        val t = DoubleArray(n) { (fs[it].ts - t0) / 1e9 }
        val mx = x.average(); val mt = t.average(); val my = fs.sumOf { it.blocks.toDouble() } / n
        var sxx = 0.0; var stt = 0.0; var sxt = 0.0; var sxy = 0.0; var sty = 0.0; var syy = 0.0
        for (i in 0 until n) {
            val a = x[i] - mx; val b = t[i] - mt; val c = fs[i].blocks - my
            sxx += a * a; stt += b * b; sxt += a * b; sxy += a * c; sty += b * c; syy += c * c
        }
        val least = tooth / 1e6 / 8
        if (stt <= 0 || sxx / n < least * least || sxt * sxt >= 0.81 * sxx * stt) return Fit(0.0, 0.0, false, my)
        val det = sxx * stt - sxt * sxt
        val g = (sxy * stt - sty * sxt) / det; val h = (sty * sxx - sxy * sxt) / det
        val s2 = maxOf(1e-9, (syy - g * sxy - h * sty) / (n - 3))
        return Fit(g, g / Math.sqrt(s2 * stt / det), true, my)
    }

    // ai: What the hold's frames say, once a batch of them is in.
    private fun decide() {
        if (since == Long.MAX_VALUE || move || moving || window.isEmpty()) return
        if (byLeaks) held() else decideByBlocks()
        snap()
    }
    private fun decideByBlocks() {
        val newest = window.last().ts
        if (window.count { it.ts > settle } < NEED) return
        val slide = fit(newest - SHORT_NS)
        // ai: a slope: the better phase is that way. In the newest frames it is a slide the pace misses, and the
        // ai: line moves a tooth (the frames say nothing of phases further off than they were); only in the longer
        // ai: span, a tilt across the phases that read, and half a tooth
        for (f in listOfNotNull(slide, fit(newest - LONG_NS))) {
            if (!f.spread || abs(f.zg) < Z) continue
            val hop = (if (f.g > 0) 1 else -1) * (if (f === slide) tooth else tooth / 2)
            line += hop; move = true
            // ai: and the pace goes the way the line had to: a TI-th of the move a second, about the last
            // ai: measurement and no further from it than its error
            val far = if (sure == Double.MAX_VALUE) PACE_MOST else sure
            pace = (pace + hop.toDouble() / TI_NS).coerceIn(anchor - far, anchor + far)
            // ai: the hold's own spacing: the line's mean period since it began on the arc's middle, to half the
            // ai: arc over the time held
            val held = (line - line0).toDouble()
            if (line0 != 0L && slots > 0 && held >= PACED_NS && wide / 2.0 / held < sure) {
                sure = wide / 2.0 / held
                anchor = held / slots / ref - 1; pace = anchor; proven = true
                say("phase: held %.0f s: the pace %.0f us a second (to %.0f)".format(held / 1e9, pace * 1e6, sure * 1e6))
            }
            return
        }
        // ai: no slope, and under HELD of the newest FELL frames (more than a hand's shake lasts) read: the hold is
        // ai: off the phases that read, where nothing slopes and nothing says which way they are
        val tail = window.takeLast(FELL)
        if (tail.count { reads(it) } < HELD * tail.size) {
            say("phase: the hold stopped reading")
            // ai: a pace only a search told, whose hold did not last: not known, and the next search tells it afresh
            if (!proven) sure = Double.MAX_VALUE
            search()
            return
        }
        if (slide != null) level = maxOf(level * 0.98, slide.mean)
    }

    // ai: The gain and the display's pace from the hold's own history. Between two readings that both showed a
    // ai: neighbour, the hold went later by the move made (h, ms) and earlier by the slide (the display's pace p
    // ai: less the line's, over the time dt), and a - b fell by that over the gain G: G (e2 - e1) - p dt = -h -
    // ai: pace dt. One such line a pair of readings, two unknowns: least squares over the pairs so far, each by
    // ai: the readings' errors, the older ones fading (FADE a pair). Where the pairs do not tell the two apart
    // ai: (a - b hardly moved), the pace alone at the gain as it is.
    private fun learn(de: Double, dt: Double, h: Double, paceMs: Double, w: Double) {
        val y = -h - paceMs * dt
        for (k in 0 until 5) fitS[k] *= FADE
        fitS[0] += w * de * de; fitS[1] += w * de * -dt; fitS[2] += w * dt * dt; fitS[3] += w * de * y; fitS[4] += w * -dt * y
        pairs++
        val det = fitS[0] * fitS[2] - fitS[1] * fitS[1]
        // ai: the gain from three pairs on (two lines through two unknowns are whatever their noise says), half
        // ai: the way to what they say at a time, and only on a pair in which a - b moved (Z of its errors): in a
        // ai: hold that stands, a - b does not move and the pairs say nothing of it
        var g = gain
        if (pairs >= 3 && de * de * w > Z * Z && det > 1e-6 * fitS[0] * fitS[2]) g = (gain + ((fitS[3] * fitS[2] - fitS[4] * fitS[1]) / det).coerceIn(GAIN_LEAST, GAIN_MOST)) / 2
        gain = g
        // ai: the pace at that gain: of -dt p = y - G de, by least squares
        val pMs = (fitS[4] - g * fitS[1]) / fitS[2]
        pace = (pMs * 1e-3).coerceIn(-PACE_MOST, PACE_MOST)
        anchor = pace
        sure = Math.max(1e-6, g * 1e-3 / Math.sqrt(fitS[2]))
        proven = true
    }

    // ai: The pilots' hold: what the newest frames hold of their neighbours, and the move it asks.
    // ai:   both neighbours in them (a top that bends: no phase is clean): later by the gain times a - b;
    // ai:   one neighbour only (the edge of a top that is flat, where a capture holds neither): out of it and a
    // ai:     step of CROSS in. Back at the same edge after standing clear of it, the two clocks' slide is the
    // ai:     step over the time it took, and the pace takes it (in the flat no reading shows a slide). At the
    // ai:     other edge straight from the step, the top is narrower than the step: its width is known, and the
    // ai:     hold goes to its middle;
    // ai:   neither: the hold stands.
    // ai: Every reading that shows a neighbour, with the one before, also tells the gain and the pace (`learn`),
    // ai: and the move allows for what the hold has slid since its frames were captured.
    private fun held() {
        val fs = window.filter { it.ts > settle && !it.r.isNaN() }
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
            val signed = fs.count { abs(it.r) >= R_FLASH && !it.r2.isNaN() && abs(it.r2) >= R_FLASH }
            if (mid < DEEP && 2 * signed < fs.size) { line += ref / 4; move = true; prevT = 0; edgeDir = 0; stood = false; if (DEBUG) say("phase: DEBUG deep in a mix (|r| %.2f): a quarter later".format(mid)) }
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
        if (prevT != 0L && side != 0 && side == prevSide && est.t > prevT) learn(e - prevE, (est.t - prevT) / 1e9, prevHop, prevPace * 1e3, 1 / (se * se + prevSe * prevSe))
        if (DEBUG) say("phase: DEBUG leaks %d frames: before %.3f (to %.3f), after %.3f (to %.3f), gain %.1f ms, pace %.0f%s".format(est.n, est.a, est.sa, est.b, est.sb, gain, pace * 1e6,
            if (flatHalf > 0) ", a flat top %.1f ms wide".format(2 * flatHalf / 1e6) else if (edgeDir != 0) ", %.1f ms in from the %s edge".format(edgeIn / 1e6, if (edgeDir > 0) "early" else "late") else ""))
        var hop = 0L
        if (sigA && sigB) {
            edgeDir = 0; flatHalf = 0
            if (abs(e) >= Z * se) hop = Math.round(gain * e * 1e6)
            else if (!balanced && sure != Double.MAX_VALUE) {
                balanced = true
                say("phase: the hold stands: %.0f%% of the picture before and %.0f%% of the one after in a capture; the pace %.0f us a second (to %s), the line %.1f ms a unit".format(
                    100 * est.a, 100 * est.b, pace * 1e6, "%.0f".format(sure * 1e6), gain))
            }
        } else if (sigA || sigB) {
            val dir = if (sigA) 1 else -1
            val edge = Math.round(gain * (if (sigA) est.a else est.b) * 1e6)
            if (edgeDir == dir && edgeClean && est.t - edgeT >= PACED_NS) {
                // ai: back at the edge it stepped in from, having stood inside the top meanwhile: the slide is
                // ai: the step (and what it is past the edge now) over the time since the step was reckoned from
                val dt = (est.t - edgeT).toDouble()
                pace = (pace + dir * (edgeIn + edge) / dt).coerceIn(-PACE_MOST, PACE_MOST)
                anchor = pace; sure = (2 * gain * Math.max(est.sa, est.sb) * 1e6) / dt; proven = true
                say("phase: the hold slid %.1f ms in %.1f s: the pace %.0f us a second (to %.0f)".format((edgeIn + edge) / 1e6, dt / 1e9, pace * 1e6, sure * 1e6))
            } else if (edgeDir == -dir && est.t > edgeT && !edgeClean) {
                // ai: the other edge, straight from a step in from the first: the top is narrower than the step,
                // ai: as wide as what was stepped, and the hold goes to its middle
                flatHalf = maxOf(bandNs, (edgeIn - edge) / 2)
                say("phase: the top is %.1f ms wide".format(2 * flatHalf / 1e6))
            } else if (edgeDir == -dir) flatHalf = 0
            val step = if (flatHalf > 0) flatHalf else CROSS_NS
            hop = dir * (edge + step)
            // ai: the step is reckoned from now where the pace was just put right (the move below allows for the
            // ai: slide since the frames); where it was not, the hold slides on as it did, and the step is
            // ai: reckoned from the frames' own time
            edgeDir = dir; edgeIn = step; edgeT = if (pace != was) now else est.t; edgeClean = false
        } else if (edgeDir != 0 && est.t > edgeT + LAND * ref) edgeClean = true
        // ai: the frames are old by the time they are read, and the hold slid on meanwhile at what the pace was
        // ai: off by: the move allows for it (the pace is right from now on)
        hop = hop.coerceIn(-ref / 2, ref / 2)
        if (hop == 0L) stood = true
        // ai: (the next pair's move is this one without that allowance: its line reckons the slide at the new pace
        // ai: from the frames' own time)
        if (side != 0) { prevE = e; prevSe = se; prevT = est.t; prevHop = hop / 1e6; prevPace = pace; prevSide = side } else prevT = 0
        if (hop != 0L || pace != was) hop += Math.round((pace - was) * (now - est.t))
        if (hop != 0L) { line += hop; move = true }
    }

    // ai: Counts the registered frames that sit at the scan's point (by their own timestamps), and moves the point.
    private fun scan(frames: List<Frame>, version: Int) {
        val t = tally.getOrPut(point) { Tally() }
        for (f in frames) {
            if (f.ts <= seen || !f.found || abs(wrap(f.ts % gridNs - point)) > 3 * bandNs) continue
            t.frames++
            if (2 * f.blocks < version) t.short++ else t.blocks += f.blocks
        }
        if (t.frames < SCAN_NEED) return
        say("phase: %.3f ms: %d frames, %.0f%% short, %.1f blocks a full frame".format(point / 1e6, t.frames, 100.0 * t.short / t.frames, if (t.frames > t.short) t.blocks.toDouble() / (t.frames - t.short) else 0.0))
        val next = (point + STEP_NS) % gridNs
        if (abs(wrap(next - start)) >= STEP_NS / 2 && tally.size <= gridNs / STEP_NS + 1) { point = next; return }
        // ai: the scan's end: the middle of the longest run, on the circle, of points within NEAR of the best point's
        // ai: share of short frames
        val pts = tally.keys.sorted()
        val share = pts.map { tally[it]!!.let { x -> x.short.toDouble() / x.frames } }
        val least = share.min()
        var best = 0; var end = -1; var run = 0
        for (k in 0 until 2 * pts.size) {
            if (share[k % pts.size] <= least + NEAR) { run++; if (run > best && run <= pts.size) { best = run; end = k } } else run = 0
        }
        val from = pts[(end - best + 1) % pts.size]
        tally.clear(); point = -1
        if (least > 1 - NEAR) { say("phase: no point reads (the best %.0f%% short)".format(100 * least)); mode = Mode.Off; return }
        base = (from + (best - 1) * STEP_NS / 2) % gridNs
        mode = Mode.Hold
        say("phase: %d of %d points read (from %.3f ms, the best %.0f%% short); held at %.3f".format(best, pts.size, from / 1e6, 100 * least, base / 1e6))
    }

    // ai: track's standing, for the log's line and the snapshot
    private val standing get() = if (now < rest) "resting" else if (searching) "searching" else if (line < 0) "no frames yet" else if (since == Long.MAX_VALUE) "placing" else "holds"
    private fun snap() {
        val e = lastLeak
        state = State(if (byLeaks) "pilots" else "blocks", standing, e?.a ?: Double.NaN, e?.b ?: Double.NaN, e?.k ?: Double.NaN,
            if (e == null) Double.NaN else Math.hypot(e.sa, e.sb), e?.n ?: 0, gain, pace * 1e6, stood, edgeDir, 2 * flatHalf / 1e6, delaysAll)
    }

    // ai: for the log, every few seconds: what is held or owed, the delays since the last call, and a pilots hold's
    // ai: newest leaks (the prefix stays as tools/phone's readers match it)
    fun line(tsNs: Long): String? {
        val d = delays; delays = 0
        return when (mode) {
            Mode.Off -> null
            Mode.Track -> "phase: track%s, %s, the pace %.0f us a second (to %s), %.1f blocks a frame, %d delays%s".format(if (flashing) " by the pilots" else "",
                standing, pace * 1e6, if (sure == Double.MAX_VALUE) "?" else "%.0f".format(sure * 1e6), level, d,
                lastLeak?.let { if (byLeaks && standing == "holds") ", %.0f%% of the picture before and %.0f%% of the one after in a capture".format(100 * it.a, 100 * it.b) else "" } ?: "")
            else -> if (base < 0 && point < 0) null else "phase: %.3f ms, %s %.3f, %d delays".format(tsNs % gridNs / 1e6, if (point >= 0) "scanning at" else "held at", (if (point >= 0) point else base) / 1e6, d)
        }
    }

    companion object {
        val DEBUG = System.getProperty("phase.debug") == "1"   // ai: tools/phasesim: the sweep's samples and a lost hold's frames
        // ai: the most the pace may be; the rest when nothing reads, and the longest it grows to
        const val PACE_MOST = 5e-3
        const val REST_NS = 20_000_000_000L
        const val REST_MOST = 320_000_000_000L
        // ai: track. A search: how long its moves go on at most and how fast (a frame's worth in 112 frames: twice as
        // ai: fast, with delays queued one a frame, told the period from half the time between passes and the hold
        // ai: after it read worse in the model), the most a step of it is (a sixteenth of a frame: what the rate
        // ai: makes where a delay lands seven captures on), how long after its moves its last frames may still come; the
        // ai: drifts tried, finely and coarsely; how much less likely than the best a drift's arc may be and still be a candidate (ln);
        // ai: the error under which a period counts as told (what the hold's moves can follow). The hold: the most
        // ai: a tooth is; the two spans of frames a slope is looked for in; the frames since the last move before
        // ai: anything is decided (and the fewest a search is fitted to); how many of its standard errors a slope
        // ai: must be; the share of a reading frame's blocks under which a frame does not read, and the share of
        // ai: the newest FELL frames that must read for the hold to stand; the least time the hold's own spacing
        // ai: is taken over; the seconds over which a move of the line is a move of the pace; the least time
        // ai: without a delay before the camera is moved about the line on purpose; the captures after which a
        // ai: delay never reported is given up; the least time between two looks of the blocks' search or hold
        // ai: at their frames (a little under the half second the feed came at, so a feed every half second is
        // ai: looked at every time). None is a monitor's: the search measures the width the tooth, the moves and
        // ai: the pace's errors come from.
        const val SEARCH_NS = 6_000_000_000L
        const val SEARCH_RATE = 1.0 / 112
        const val SEARCH_STEPS = 16.0
        const val SEARCH_LATE = 1_500_000_000L
        const val DRIFT_STEP = 25e-6
        const val DRIFT_COARSE = 100e-6
        const val NATS = 2.0
        const val DRIFT_ERR = 0.6e-3
        const val TOOTH_NS = 1_000_000L
        const val SHORT_NS = 3_000_000_000L
        const val LONG_NS = 12_000_000_000L
        const val NEED = 40
        const val Z = 3.0
        const val HELD = 0.6
        const val FELL = 120
        const val PACED_NS = 1_000_000_000L
        const val TI_NS = 6e9
        const val DITHER_NS = 500_000_000L
        const val LOST = 60
        const val LOOK_NS = 450_000_000L
        // ai: the pilots: |r| at which a reading's sign is not noise; the newest frames with a reading that say
        // ai: whether the frames flash, and the pairs of them needed; the least error a reading is given (its own
        // ai: at LIZARD-512 is about 0.02 over half the blocks); the frames an estimate of the leaks needs; the
        // ai: median |r| under which the frames are deep in a mix; the ms the line moves a unit of a - b before
        // ai: any move has measured it (under twice the least a display's own is, so a first move cannot grow the
        // ai: error: a window of 9 ms that changes pictures at one instant has 9, the S26's arithmetic about 20),
        // ai: and the least and most a measurement may say; the step in from the edge of a flat top; the span of
        // ai: the newest frames the leaks are read from; the captures a delay takes to land; the share of a
        // ai: neighbour under which a capture is taken to hold none of it; what a pair of readings keeps of its
        // ai: weight in the fit of the gain and the pace with each pair after it.
        const val R_FLASH = 0.25
        const val FLASH_FRAMES = 60
        const val FLASH_PAIRS = 12
        const val R_NOISE_MIN = 0.005
        const val L_NEED = 12
        const val DEEP = 0.35
        const val GAIN_MS = 12.0
        const val GAIN_LEAST = 2.0
        const val GAIN_MOST = 40.0
        const val CROSS_NS = 2_000_000L
        const val RECENT_NS = 1_000_000_000L
        const val LAND = 8
        const val LEAK = 0.02
        const val FADE = 0.85
        // ai: hold and scan: the grid (a 60 Hz display's refresh), the band (a capture this far early is delayed
        // ai: back), a scan's step, the frames counted at a point, and how far over the best point's share of short
        // ai: frames a point may be and still count as reading as well
        const val GRID_NS = 16_666_667L
        const val BAND_NS = 250_000L
        const val STEP_NS = 1_000_000L
        const val SCAN_NEED = 30
        const val NEAR = 0.1
    }
}
