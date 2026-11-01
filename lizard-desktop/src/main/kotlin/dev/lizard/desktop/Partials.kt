package dev.lizard.desktop

import dev.lizard.Session
import dev.lizard.ui.Partial
import dev.lizard.ui.partialsOf

// ai: Transfers begun and not finished (2026-10-25, the PC receiver): the receive session's list of its store's folders
// ai: under Native.data (dev.lizard.ui Partial, partialsOf, as the Android app reads its own)
object Partials {
    fun list(): List<Partial> = partialsOf(Session.partials(Native.data.path))
}
