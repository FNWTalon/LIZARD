// Why "decoded N with new data" fell.
//
// It falls for unrelated reasons that look identical on that one line, and the line alone sent a
// four-minute frozen screen (research/rig/stats.jsonl, a run at foundShare 100%, bad 0, every block of
// every frame decoding, freshPerFrame 0 throughout) looking like a codec problem. The three
// causes are already separable from numbers the receiver computes:
//
//   nothing registers                 foundShare low        aim, distance, blur
//   registers, no block survives FEC  empty ~= processed    the channel
//   registers, every block a repeat   repeat ~= processed   the picture is not changing
//
// Every result increments exactly one of withNew, repeat and empty (recv.mjs onResult), so
// decodedFps + repeatFps + emptyFps === processedFps and the shares below are exact.
//
// Pure, and takes only a stats row, so it runs over recorded rows in a test (scripts/exp/verdict_check.mjs).

// Treated as stalled below this share of processed frames carrying something new. A 5 s window at
// 30 fps holds ~150 frames, so this still allows a few genuine new frames before it speaks up.
const STALLED = 0.05;

// ai: The receiver hears nothing from the sender but its light (2026-09-26), so which end stopped is not said
// ai: here: the sender's own line beside the row says whether it was painting.
/**
 * @param {object} stats   the row recv.mjs builds each second
 * @returns {{ level: "ok"|"warn", text: string } | null}  null when nothing is wrong
 */
export function verdict(stats) {
  const proc = stats.processedFps ?? 0;
  if (proc < 0.5) return { level: "warn", text: "no frames are reaching the decoders" };
  if ((stats.decodedFps ?? 0) >= STALLED * proc) return null;

  if ((stats.foundShare ?? 0) < 0.5)
    return { level: "warn", text: "NO NEW DATA: the code is not being registered. Aim, distance, focus or glare." };

  if ((stats.repeatFps ?? 0) >= (stats.emptyFps ?? 0))
    return { level: "warn", text: "NO NEW DATA: every block decodes but the picture on screen is not changing." };

  return { level: "warn", text: "NO NEW DATA: the code registers but no block survives FEC. The channel, not the sender." };
}
