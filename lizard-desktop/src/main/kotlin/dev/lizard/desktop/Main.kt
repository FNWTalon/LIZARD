package dev.lizard.desktop

import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.WindowPlacement
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import java.awt.KeyEventDispatcher
import java.awt.KeyboardFocusManager
import java.awt.event.KeyEvent

// ai: The desktop sender (2026-10-04): the web sender's page as a window (lizard-web/send.html), the
// ai: column on the left and the code area on the right, the code drawn there by the native presenter. LIZ_SENDER_TEST
// ai: (SenderTest.kt) runs the integration check instead of waiting for a hand.
fun main() {
    // ai: AWT erases nothing under the presenter's pictures (the code area's canvas is the swapchain's)
    System.setProperty("sun.awt.noerasebackground", "true")
    val test = SenderTest.parse(System.getenv("LIZ_SENDER_TEST"))
    if (test != null) {
        Prefs.forget()
        Thread.setDefaultUncaughtExceptionHandler { _, e -> SenderTest.fail("${e.javaClass.simpleName}: ${e.message}") }
    }
    val s = SendState()
    try { Native.load() } catch (e: Throwable) { s.loadError = "The native library could not be loaded: ${e.message}" }
    test?.apply(s)

    application {
        val ws = rememberWindowState(width = 1280.dp, height = 800.dp)
        var before by remember { mutableStateOf(WindowPlacement.Floating) }
        Window(onCloseRequest = { s.stop(); exitApplication() }, state = ws, title = "LIZARD sender", icon = rememberVectorPainter(Icons.mark)) {
            // ai: the frame white before Compose's first frame (2026-10-07): AWT's default grey showed until Skia drew
            remember { window.background = java.awt.Color.WHITE; window.contentPane.background = java.awt.Color.WHITE }
            // ai: F11 full screen and back, Escape out of it, wherever the focus is (the canvas takes none)
            DisposableEffect(Unit) {
                val keys = KeyEventDispatcher { e ->
                    if (e.id != KeyEvent.KEY_PRESSED) false
                    else when {
                        e.keyCode == KeyEvent.VK_ESCAPE && s.askBrightness -> { s.answerBrightness(false); true }
                        e.keyCode == KeyEvent.VK_F11 -> { s.toggleFull(); true }
                        e.keyCode == KeyEvent.VK_ESCAPE && s.full -> { s.toggleFull(); true }
                        else -> false
                    }
                }
                KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(keys)
                onDispose { KeyboardFocusManager.getCurrentKeyboardFocusManager().removeKeyEventDispatcher(keys) }
            }
            // ai: (the presenter keeps the window above the others while full screen: presenter.cpp applyBypass)
            LaunchedEffect(s.full) {
                if (s.full && ws.placement != WindowPlacement.Fullscreen) { before = ws.placement; ws.placement = WindowPlacement.Fullscreen }
                else if (!s.full && ws.placement == WindowPlacement.Fullscreen) ws.placement = before
            }
            // ai: full screen left by the window manager's own means: once the window was full screen and is no longer
            // ai: (2026-10-04, on a real screen: reacting to "not full screen" undid a full screen asked before the
            // ai: manager had made it, the bypass on, then off, the column back)
            var wasFull by remember { mutableStateOf(false) }
            LaunchedEffect(ws.placement) {
                if (ws.placement == WindowPlacement.Fullscreen) wasFull = true
                else if (wasFull) { wasFull = false; if (s.full) s.toggleFull() }
            }
            SendScreen(s, window)
            if (test != null) LaunchedEffect(Unit) { test.run(s) }
        }
    }
}
