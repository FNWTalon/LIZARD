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
import java.awt.EventQueue
import java.awt.KeyEventDispatcher
import java.awt.KeyboardFocusManager
import java.awt.event.InputEvent
import java.awt.event.KeyEvent

// ai: The desktop sender (2026-10-04): the web sender's page as a window (lizard-web/send.html), the
// ai: column on the left and the code area on the right, the code drawn there by the native presenter. LIZ_SENDER_TEST
// ai: (SenderTest.kt) runs the integration check instead of waiting for a hand.
// ai: args: files and folders to send, added to the selection (a terminal, a file manager's Open with through the
// ai: package's desktop entry, Exec %F); a sender already running takes them instead (Instance.kt), and this launch ends.
fun main(args: Array<String>) {
    // ai: AWT erases nothing under the presenter's pictures (the code area's canvas is the swapchain's)
    System.setProperty("sun.awt.noerasebackground", "true")
    val test = SenderTest.parse(System.getenv("LIZ_SENDER_TEST"))
    if (test != null) {
        Prefs.forget()
        Thread.setDefaultUncaughtExceptionHandler { _, e -> SenderTest.fail("${e.javaClass.simpleName}: ${e.message}") }
    }
    val given = args.map { java.io.File(it).absoluteFile }.filter { it.exists() }
    if (test == null && given.isNotEmpty() && Instance.handOff(given)) return
    val s = SendState()
    try { Native.load() } catch (e: Throwable) { s.loadError = "The native library could not be loaded: ${e.message}" }
    test?.apply(s)
    if (test == null) {
        Instance.listen { files -> EventQueue.invokeLater { if (s.test) s.chooseTest(false); s.pick(files); s.raise?.invoke() } }
        if (given.isNotEmpty()) { if (s.test) s.chooseTest(false); s.pick(given) }
    }

    application {
        val ws = rememberWindowState(width = 1280.dp, height = 800.dp)
        var before by remember { mutableStateOf(WindowPlacement.Floating) }
        Window(onCloseRequest = { s.close(); exitApplication() }, state = ws, title = "LIZARD sender", icon = rememberVectorPainter(Icons.mark)) {
            // ai: the frame white before Compose's first frame (2026-10-07): AWT's default grey showed until Skia drew
            remember { window.background = java.awt.Color.WHITE; window.contentPane.background = java.awt.Color.WHITE }
            // ai: Keys wherever the focus is (the canvas takes none), as video players have them (2026-10-23; F11 and Escape
            // ai: before): Space or K play and pause (SendState.playPause), F or F11 full screen and back, Escape out of it,
            // ai: Ctrl+O add files where files are sent. A key held down acts once; the keys taken are taken whole (press,
            // ai: typed, release), so Space on a focused button is the hotkey and not that button's press. While the
            // ai: brightness tip is open, Escape alone (its Cancel).
            DisposableEffect(Unit) {
                val held = HashSet<Int>()
                val taken = HashSet<Int>()
                val mods = InputEvent.CTRL_DOWN_MASK or InputEvent.ALT_DOWN_MASK or InputEvent.META_DOWN_MASK or InputEvent.SHIFT_DOWN_MASK
                val keys = KeyEventDispatcher { e ->
                    when (e.id) {
                        KeyEvent.KEY_RELEASED -> { held.remove(e.keyCode); taken.remove(e.keyCode) }
                        KeyEvent.KEY_TYPED -> taken.isNotEmpty() && e.keyChar in " kKfF"
                        KeyEvent.KEY_PRESSED -> {
                            val first = held.add(e.keyCode)
                            val m = e.modifiersEx and mods
                            val act: (() -> Unit)? = when {
                                s.askBrightness -> if (e.keyCode == KeyEvent.VK_ESCAPE) ({ s.answerBrightness(false) }) else null
                                e.keyCode == KeyEvent.VK_F11 -> ({ s.toggleFull() })
                                e.keyCode == KeyEvent.VK_ESCAPE && s.full -> ({ s.toggleFull() })
                                m == 0 && (e.keyCode == KeyEvent.VK_SPACE || e.keyCode == KeyEvent.VK_K) -> ({ s.playPause() })
                                m == 0 && e.keyCode == KeyEvent.VK_F -> ({ s.toggleFull() })
                                m == InputEvent.CTRL_DOWN_MASK && e.keyCode == KeyEvent.VK_O -> ({ if (!s.test) s.chooseFile() })
                                else -> null
                            }
                            if (act != null) { taken += e.keyCode; if (first) act() }
                            act != null
                        }
                        else -> false
                    }
                }
                KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(keys)
                onDispose { KeyboardFocusManager.getCurrentKeyboardFocusManager().removeKeyEventDispatcher(keys) }
            }
            // ai: Full screen as other apps have it (2026-10-23): the window manager's own (_NET_WM_STATE_FULLSCREEN,
            // ai: Native.windowFullscreen), which gives the window back as it was, maximized or not; the app follows the
            // ai: manager's word, read back at each resize of the window, so its own shortcut works too. Off X11,
            // ai: Compose's placement. Until then: Compose's placement on X11, and a leave through Floating then
            // ai: Maximized once the bounds held still, since Compose cannot set a window maximized again.
            s.maximize = { ws.placement = WindowPlacement.Maximized }
            s.raise = { window.toFront() }
            s.fullscreen = { on ->
                val top = runCatching { Native.windowOf(s.canvas) }.getOrDefault(0L)
                if (!runCatching { Native.windowFullscreen(top, on) }.getOrDefault(false)) {
                    if (on) { before = ws.placement; ws.placement = WindowPlacement.Fullscreen } else ws.placement = before
                }
            }
            DisposableEffect(Unit) {
                val resized = object : java.awt.event.ComponentAdapter() {
                    override fun componentResized(e: java.awt.event.ComponentEvent) {
                        val top = runCatching { Native.windowOf(s.canvas) }.getOrDefault(0L)
                        if (top != 0L) s.fullFromManager(Native.windowIsFullscreen(top))
                    }
                }
                window.addComponentListener(resized)
                onDispose { window.removeComponentListener(resized) }
            }
            SendScreen(s, window)
            if (test != null) LaunchedEffect(Unit) { test.run(s) }
        }
    }
}
