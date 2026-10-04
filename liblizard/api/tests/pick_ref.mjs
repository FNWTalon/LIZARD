// ai: The web sender's picker as a table, for the C API's test (tests/roundtrip.c) to hold liz_room_for and liz_pick
// ai: to (2026-10-03): ROOM_FOR for every block count and ring, then pickVersion on the room lizard-web/send.mjs gives two
// ai: codes or one in a w x h area (the width over the codes and their gap, against the height). Prints the table.
// ai:   node api/tests/pick_ref.mjs [out]   (the table to out, else to stdout; needs the codec's wasm built: ./build.sh)
import { init } from "../../sim/ob.mjs";

await init();
const { ROOM_FOR, pickVersion, MODULES, OB_QUIET } = await import("../../sim/lizard_pick.mjs");
const out = [];
for (let r = 0; r < 4; r++) for (let b = 1; b <= 128; b++) out.push(`room ${b} ${r} ${ROOM_FOR(8 * b, r).toPrecision(17)}`);
for (const codes of [1, 2])
  for (let r = 0; r < 4; r++)
    for (const top of [128, 52, 8])
      for (let w = 200; w <= 4000; w += 173)
        for (let h = 200; h <= 2200; h += 211) {
          const gf = codes > 1 ? ((codes - 1) * 12) / (MODULES(256, r) + 2 * OB_QUIET) : 0;
          const room = Math.max(64, Math.min(w / (codes + gf), h));
          out.push(`pick ${w} ${h} ${codes} ${r} ${top} ${pickVersion(room, 8 * top, r).subch / 8}`);
        }
if (process.argv[2]) (await import("node:fs")).writeFileSync(process.argv[2], out.join("\n") + "\n");
else console.log(out.join("\n"));
