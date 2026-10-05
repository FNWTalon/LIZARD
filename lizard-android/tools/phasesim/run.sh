#!/bin/bash
# ai: PhaseLock.kt compiled with Sim.kt by the Kotlin compiler Gradle already fetched for the app (its cache), and run:
# ai: the lock's logic against a model and against the phone's recorded searches (searches.txt), no device and no new
# ai: dependency. Exit 1 when a case fails: 8 do as found (6 since the signed pilots, 2026-10-01; 8 since the rate-switch
# ai: cases, 2026-10-04): the blocks arm at the box's 2.2 ms window, six pilots rows on the box model, and the blocks arm
# ai: on the measured model's rate switch (it searches again after the switch); none of the pilots' measured rows.
# ai: About four minutes.
# ai: `run.sh replay`: the recorded searches only, 5 s. `run.sh series <file>`: a replay's recorded series (the text
# ai: scripts/exp/replay_phase.py --series writes) through the lock itself, open loop, its lines as they come (2026-10-04).
set -u
H=$(cd "$(dirname "$0")" && pwd); DATA=${PHASESIM_DIR:-$H}; C=${GRADLE_USER_HOME:-$HOME/.gradle}/caches/modules-2/files-2.1; K=org.jetbrains.kotlin
jar() { find "$C/$1" -name "$2" 2>/dev/null | grep -v sources | head -1; }
KC=$(jar $K/kotlin-compiler-embeddable "kotlin-compiler-embeddable-2.1.20.jar"); KS=$(jar $K/kotlin-stdlib "kotlin-stdlib-2.1.20.jar")
[ -n "$KC" ] && [ -n "$KS" ] || { echo "no Kotlin 2.1.20 compiler in the Gradle cache (build the app once: ./build.sh apk)" >&2; exit 2; }
CP="$KC:$KS:$(jar $K/kotlin-script-runtime '*.jar'):$(jar $K/kotlin-reflect/1.6.10 '*.jar'):$(jar $K/kotlin-daemon-embeddable '*.jar'):$(find $C/org.jetbrains.intellij.deps -name '*.jar' | grep -v sources | tr '\n' ':')$(jar org.jetbrains.kotlinx 'kotlinx-coroutines-core-jvm*.jar'):$(jar org.jetbrains/annotations 'annotations-*.jar')"
OUT=$(mktemp -d)
java -cp "$CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -cp "$KS" "${PHASE_LOCK:-$H/../../app/src/main/java/dev/lizard/receiver/PhaseLock.kt}" "${PHASE_SIM:-$H/Sim.kt}" -d "$OUT" 2>&1 | grep -i "error:\|caused by" | head -8
java -Dphasesim.dir="$DATA" -cp "$OUT:$KS" SimKt "$@"; rc=$?
rm -rf "${OUT:?}"
exit $rc
