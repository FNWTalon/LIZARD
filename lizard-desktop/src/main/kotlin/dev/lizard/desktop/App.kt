package dev.lizard.desktop

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.ApplicationScope
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.WindowPlacement
import androidx.compose.ui.window.rememberWindowState
import java.awt.KeyEventDispatcher
import java.awt.KeyboardFocusManager
import java.awt.event.InputEvent
import java.awt.event.KeyEvent
import java.io.File

// ai: The app's screens (2026-10-25, the PC receiver: the desktop app is LIZARD, the web's three pages and the Android
// ai: app's screens in one window): Home, Send and Receive, the window titled LIZARD on each.
enum class Screen { Home, Send, Receive }

// ai: The app (one window): the screen shown, the sender's state and the receiver's, and what is the window's. Between
// ai: screens (go): leaving Send stops the send (SendState.leave), leaving Receive stops the camera and closes it (the
// ai: session and a transfer partly in kept, ReceiveState.leave); entering Receive starts the camera, as the Android
// ai: app's go(). Files handed in by a later launch (Instance) go to Send.
class App {
    var screen by mutableStateOf(Screen.Home)
    val send = SendState()
    val recv = ReceiveState(this)
    // ai: why the native library did not load ("" it did): the sender's line and the receiver's say it
    var loadError = ""
        set(v) { field = v; send.loadError = v }
    val about: String get() = send.about

    fun go(to: Screen) {
        if (to == screen) return
        when (screen) {
            Screen.Send -> send.leave()
            Screen.Receive -> recv.leave()
            Screen.Home -> {}
        }
        screen = to
        when (to) {
            Screen.Home -> recv.refreshLists()
            Screen.Receive -> recv.enter()
            Screen.Send -> {}
        }
    }

    // ai: paths handed in (a later launch's, Instance.kt): to Send, added to its selection, the window raised
    fun handIn(files: List<File>) {
        go(Screen.Send)
        if (send.test) send.chooseTest(false)
        send.pick(files)
        send.raise?.invoke()
    }

    // ai: the window put away (LizardWindow sets it): the receiver's close may wait seconds on a filing under way
    var hide: (() -> Unit)? = null
    // ai: the sender first (its presenter stopped while the window still shows its canvas), then the window hidden, then
    // ai: the receiver, whose close waits for its files' work (ReceiveState.close)
    fun close() { send.close(); runCatching { hide?.invoke() }; recv.close() }
}

// ai: The window: the screen shown, the sender's keys while it is Send's, full screen as the sender asks it; the test
// ai: hooks run inside it (SenderTest.kt, ReceiverTest.kt).
@Composable
fun ApplicationScope.LizardWindow(app: App, sendTest: SenderTest?, recvTest: ReceiverTest?) {
    val s = app.send
    val ws = rememberWindowState(width = 1280.dp, height = 800.dp)
    var before by remember { mutableStateOf(WindowPlacement.Floating) }
    Window(onCloseRequest = { app.close(); exitApplication() }, state = ws, title = "LIZARD", icon = rememberVectorPainter(Icons.mark)) {
        // ai: the frame white before Compose's first frame (2026-10-07): AWT's default grey showed until Skia drew
        remember { window.background = java.awt.Color.WHITE; window.contentPane.background = java.awt.Color.WHITE }
        // ai: Keys wherever the focus is (the canvas takes none), as video players have them (2026-10-23; F11 and Escape
        // ai: before): Space or K play and pause (SendState.playPause), F or F11 full screen and back, Escape out of it,
        // ai: Ctrl+O (Command+O on macOS) add files where files are sent. A key held down acts once; the keys taken are taken whole (press,
        // ai: typed, release), so Space on a focused button is the hotkey and not that button's press. While the
        // ai: brightness tip is open, Escape alone (its Cancel). Send's alone (2026-10-25): Home and Receive take no key.
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
                            app.screen != Screen.Send -> null
                            s.askBrightness -> if (e.keyCode == KeyEvent.VK_ESCAPE) ({ s.answerBrightness(false) }) else null
                            e.keyCode == KeyEvent.VK_F11 -> ({ s.toggleFull() })
                            e.keyCode == KeyEvent.VK_ESCAPE && s.full -> ({ s.toggleFull() })
                            m == 0 && (e.keyCode == KeyEvent.VK_SPACE || e.keyCode == KeyEvent.VK_K) -> ({ s.playPause() })
                            m == 0 && e.keyCode == KeyEvent.VK_F -> ({ s.toggleFull() })
                            m == (if (Native.mac) InputEvent.META_DOWN_MASK else InputEvent.CTRL_DOWN_MASK) && e.keyCode == KeyEvent.VK_O -> ({ if (!s.test) s.chooseFile() })
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
        // ai: manager's word, read back at each resize of the window, so its own shortcut works too. Windows
        // ai: (2026-10-24): the same, the window's frame off over its monitor (Native.windowFullscreen); macOS,
        // ai: Compose's placement. Until then: Compose's placement on X11, and a leave through Floating then
        // ai: Maximized once the bounds held still, since Compose cannot set a window maximized again. The window is
        // ai: found through the sender's canvas, which has none on Home and Receive (0: nothing read there).
        s.maximize = { ws.placement = WindowPlacement.Maximized }
        s.raise = { window.toFront() }
        app.hide = { window.isVisible = false }
        s.fullscreen = { on ->
            val top = runCatching { Native.windowOf(s.canvas) }.getOrDefault(0L)
            if (!runCatching { Native.windowFullscreen(top, on) }.getOrDefault(false)) {
                if (on) { before = ws.placement; ws.placement = WindowPlacement.Fullscreen } else ws.placement = before
            }
        }
        DisposableEffect(Unit) {
            val resized = object : java.awt.event.ComponentAdapter() {
                override fun componentResized(e: java.awt.event.ComponentEvent) {
                    if (app.screen != Screen.Send) return
                    val top = runCatching { Native.windowOf(s.canvas) }.getOrDefault(0L)
                    if (top != 0L) s.fullFromManager(Native.windowIsFullscreen(top))
                }
            }
            window.addComponentListener(resized)
            onDispose { window.removeComponentListener(resized) }
        }
        MaterialTheme(colorScheme = LizardColours, typography = LizardType) {
            when (app.screen) {
                Screen.Home -> HomeScreen(app)
                Screen.Send -> SendScreen(s, window, onSwap = { app.go(Screen.Receive) }, onHome = { app.go(Screen.Home) })
                Screen.Receive -> ReceiveScreen(app)
            }
        }
        if (sendTest != null) LaunchedEffect(Unit) { sendTest.run(s) }
        if (recvTest != null) LaunchedEffect(Unit) { recvTest.run(app) }
    }
}
