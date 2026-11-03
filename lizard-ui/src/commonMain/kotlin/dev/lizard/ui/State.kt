package dev.lizard.ui

import dev.lizard.readout.Readout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonObject
import java.io.File
import kotlin.math.floor

// ai: What both apps read out of the receive session's JSON (2026-10-31; each app's own parsing before, the desktop's
// ai: kotlinx and the Android app's org.json): the stats a quarter second (liblizard/core/rx/session.h stats(): the
// ai: receiver's fields, the session's phase, its camera, its runs), the store's unfinished transfers (Session.partials).
// ai: Pure: no drawing here.

private val json = Json { ignoreUnknownKeys = true }
fun parse(s: String): JsonObject? = runCatching { json.parseToJsonElement(s).jsonObject }.getOrNull()
fun JsonObject.num(k: String) = (this[k] as? JsonPrimitive)?.doubleOrNull ?: 0.0
fun JsonObject.str(k: String) = (this[k] as? JsonPrimitive)?.contentOrNull ?: ""
fun JsonObject.flag(k: String) = (this[k] as? JsonPrimitive)?.booleanOrNull ?: false
fun JsonObject.obj(k: String) = this[k] as? JsonObject
fun JsonObject.objs(k: String) = (this[k] as? JsonArray)?.mapNotNull { it as? JsonObject }

// ai: The receiver's phase as a screen shows it (the session's "phase" and "why"; Starting while an app's own start
// ai: order runs before the session's start, `opening`)
sealed interface RxPhase {
    data object Idle : RxPhase
    data object Loading : RxPhase
    data object Starting : RxPhase
    data object On : RxPhase
    data class Error(val why: String) : RxPhase
}

fun rxPhaseOf(j: JsonObject?, opening: Boolean = false): RxPhase = when (j?.str("phase")) {
    "starting" -> RxPhase.Starting
    "loading" -> RxPhase.Loading
    "on" -> RxPhase.On
    "error" -> RxPhase.Error(j.str("why"))
    else -> if (opening) RxPhase.Starting else RxPhase.Idle
}

// ai: The stats' fields the screens read (liblizard/core/rx/receiver.h stats() and receiver.cpp name them)
fun rxOf(j: JsonObject?): Readout.Rx {
    if (j == null) return Readout.Rx()
    val f = j.obj("file")
    return Readout.Rx(state = j.str("state"), error = j.str("error"), goodputKBs = j.num("goodputKBs"), foundShare = j.num("foundShare"),
        bandVersion = j.num("bandVersion").toInt(), fps = j.obj("word")?.num("fps")?.toInt() ?: 0,
        hasFile = f != null, name = f?.str("name").orEmpty().ifEmpty { "a file" }, size = f?.num("size")?.toLong() ?: 0,
        received = f?.num("received")?.toLong() ?: 0, verified = f?.flag("verified") ?: false,
        sent = f?.num("sent")?.toLong() ?: 0, sentIn = f?.num("sentIn")?.toLong() ?: 0, secs = f?.num("secs") ?: 0.0)
}

// ai: Save replays' runs (the stats' "replays", session.h): a run's .lizard of the newest frames the decoder takes
enum class RunState { Recording, Ending, Ready, Failed }
data class Run(val run: String, val file: File, val state: RunState, val frames: Long = 0, val bytes: Long = 0, val why: String = "")

// ai: the runs the stats list, null where they say nothing of them
fun runsOf(j: JsonObject?): List<Run>? = j?.objs("replays")?.map { o ->
    val st = when (o.str("state")) { "recording" -> RunState.Recording; "saving" -> RunState.Ending; "ready" -> RunState.Ready; else -> RunState.Failed }
    Run(o.str("run"), File(o.str("path")), st, o.num("frames").toLong(), o.num("bytes").toLong(), o.str("error"))
}

// ai: a run's line in Developer Tools
fun runLine(r: Run): String = when (r.state) {
    RunState.Recording -> "${r.run}: recording"
    RunState.Ending -> "${r.run}: saving"
    RunState.Ready -> "${r.run}: ${r.frames} frames, ${String.format(java.util.Locale.ROOT, "%.0f", r.bytes / 1e6)} MB" + if (r.why.isEmpty()) "" else " (${r.why})"
    RunState.Failed -> "${r.run}: ${r.why}"
}

// ai: Transfers begun and not finished (2026-10-22): the receive session's store, one folder a transfer by its root's
// ai: first 16 hex digits with its header and meta.json; Session.partials lists them, newest first. A receiver resumes
// ai: one when its file is in the light again; Home lists them to continue or delete. The web keeps the same in its
// ai: origin's OPFS (lizard-web/library.mjs unfinished).
data class Partial(val id: String, val name: String, val length: Long, val fraction: Double, val root: String, val at: Long)

fun partialsOf(listing: String): List<Partial> = runCatching {
    (json.parseToJsonElement(listing) as? JsonArray).orEmpty().mapNotNull { e ->
        (e as? JsonObject)?.let { o -> Partial(o.str("id"), o.str("name"), o.num("length").toLong(), o.num("fraction"), o.str("root"), o.num("at").toLong()) }
    }
}.getOrDefault(emptyList())

// ai: One hint, the first that holds, while the camera reads a word (2026-10-25, both apps' since 2026-10-31). A phone
// ai: as the camera (cam/phone.cpp) or an upload, none: their timing is not this camera's. A camera the phase lock does not hold reads a sender at most half its rate without its captures spanning two
// ai: pictures (the web's rule), so its rate (the one it agreed to at start, else `modeFps`) under twice the word's
// ai: names the most the sender should paint.
fun hintOf(j: JsonObject?, on: Boolean, phone: Boolean = false, upload: Boolean = false, locked: Boolean = false, modeFps: Double = 0.0): String {
    if (!on || j == null || upload || phone) return ""
    val cam = j.obj("camera")
    val wf = j.obj("word")?.num("fps") ?: 0.0
    if (j.num("bandVersion") <= 0 || wf <= 0 || cam == null) return ""
    val camFps = cam.num("agreedFps").takeIf { it > 0 } ?: modeFps
    if (!locked && camFps > 0 && camFps < 2 * wf) {
        val n = floor(camFps / 2).toInt()
        if (n >= 1) return "Set the sender to at most $n fps for this camera."
    }
    return ""
}
