// How many decode workers to run, and when to change it.
//
// The pool exists because one worker may not keep up with the camera. It is not free: each worker
// holds its own wasm instance and its own frame buffers, and a phone pays for every one of them in
// heat. So the rule is that a worker is added only when another one could carry data this pool is
// missing, and given back once it is not.
//
// Pure, so the policy can be exercised without a camera: the page owns the workers, this owns the
// decision. See scripts/exp/pool_check.mjs.

export const DEFAULTS = {
  // Frames lost to busy workers before one is added. A single slow frame is not a reason.
  threshold: 4,
  // Seconds of spare capacity before one is given back. Far slower to give up than to take, so a
  // periodic hitch cannot start the pool oscillating.
  slackSecs: 10,
  // Above this share of repeats the camera is outrunning the sender, and a frame lost to busy
  // workers was a duplicate of one already decoded. Another worker would decode it again: all of
  // the heat, none of the data.
  repeatCeiling: 0.15,
  // Frames a second still lost to busy workers while a second counts as slack. Demanding none at
  // all is what made this a ratchet: in the regime the pool really runs in, just about keeping up
  // and dropping the odd frame, one of the ten seconds always had a skip in it, the count went
  // back to zero, and one hard stretch pinned the pool at its ceiling for the rest of the session.
  slackSkips: 2,
};

export class PoolPolicy {
  constructor(opts = {}) { Object.assign(this, DEFAULTS, opts); this.reset(); }
  reset() { this.pressure = 0; this.slack = 0; }

  // A frame was lost because every worker was busy. Returns true to add one.
  //
  // allReady false means a worker is still starting: capacity is already on its way. Counting
  // through that boot banks the whole threshold and spends it the instant the worker arrives,
  // which walks the pool to its ceiling a step at a time without ever testing whether the last
  // step was enough.
  lost({ allReady, size, ceiling, repeatShare }) {
    if (!allReady) return false;
    if (repeatShare > this.repeatCeiling) { this.pressure = 0; return false; }
    if (++this.pressure < this.threshold || size >= ceiling) return false;
    this.pressure = 0; this.slack = 0;
    return true;
  }
  // There WAS a rule here that refused to grow into a pool size already seen to carry no more frames a second
  // than the size below it. It is gone, and the reason is worth keeping. The figures behind it were averaged
  // across sessions at different crops, distances and temperatures, and the code then made the same mistake in
  // the small: it compared a rate measured at one moment with one measured minutes earlier under a different
  // crop, and nothing ever invalidated either. Once two neighbouring sizes had both been measured and come out
  // alike, growth was refused for the rest of the run. A pool sat at 2 with a ceiling of 8 and dropped two
  // thirds of its frames for thirty seconds (build/captures/run-2026-09-21T20-04-09-535Z). repeatCeiling above
  // already answers the question that rule was for, and it answers it from the CURRENT window.

  // One second passed. Returns true to give a worker back.
  //
  // Without this the pool only ratchets up, and one hard stretch leaves it wide for the rest of
  // the run. A second that lost nothing to busy workers and ended with a worker idle is slack.
  tick({ busySkips, size, hasIdle, auto }) {
    if (!busySkips) this.pressure = 0;
    this.slack = busySkips <= this.slackSkips && size > 1 && hasIdle ? this.slack + 1 : 0;
    if (this.slack < this.slackSecs || !auto) return false;
    this.slack = 0; this.pressure = 0;
    return true;
  }
}
