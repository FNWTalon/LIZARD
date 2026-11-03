package dev.lizard.desktop

import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.input.pointer.PointerIcon
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
import dev.lizard.ui.*
import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsDraggedAsState
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest

// ai: The desktop's target size: the web's under a mouse (ui.css pointer:fine), buttons, fields and icon buttons 40 dp,
// ai: where the phone's are 48 (the screens are ../lizard-ui's, dev.lizard.ui, 2026-10-31)
val TARGET = 40.dp

// ai: the pointer over a column's edge that drags sideways: AWT's left-right resize cursor
val RESIZE = PointerIcon(java.awt.Cursor(java.awt.Cursor.E_RESIZE_CURSOR))

// ai: The app (one window): the screen shown, the sender's state and the receiver's, and what is the window's. Between
// ai: screens (go): leaving Send stops the send (SendState.leave), leaving Receive stops the camera and closes it (the
// ai: session and a transfer partly in kept, ReceiveState.leave); entering Receive starts the camera, as the Android
// ai: app's go(). Files handed in by a later launch (Instance) go to Send.
class App {
    var screen by mutableStateOf(Screen.Home)
    val send = SendState()
    val recv = ReceiveState(this)
    val home = DesktopHome(this)
    val folds = DesktopFolds(this)
    val column = DesktopColumn()
    val receive = DesktopReceive(this)
    val sendUi = DesktopSend(this)
    // ai: why the native library did not load ("" it did): the sender's line and the receiver's say it
    var loadError = ""
        set(v) { field = v; send.loadError = v }
    val about: String get() = send.about

    fun go(to: Screen) {
        if (to == screen) return
        when (screen) {
            Screen.Send -> send.leave()
            Screen.Receive -> recv.leave()
            else -> {}
        }
        screen = to
        when (to) {
            Screen.Home -> recv.refreshLists()
            Screen.Receive -> recv.enter()
            else -> {}
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
    // ai: the window itself (LizardWindow sets it): the owner of Receive's dialogs (Chooser), which has no canvas
    var frame: java.awt.Frame? = null
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
        remember { window.background = java.awt.Color.WHITE; window.contentPane.background = java.awt.Color.WHITE; if (app.loadError.isEmpty()) whiten(window); app.frame = window; app.sendUi.window = window; s.frame = window }
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
                            s.askExclusive -> if (e.keyCode == KeyEvent.VK_ESCAPE) ({ s.answerExclusive(false) }) else null
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
        // ai: found through itself (through the sender's canvas until 2026-11-02, which now has a window only while codes
        // ai: play). Its other resizes on Send are resizes by hand (SendState.resizing: a run pauses).
        // ai: the window's place kept as a monitor is taken for the code (the window manager moves it off that monitor)
        // ai: and put back once it is given back (SendState.leased)
        var placeWas by remember { mutableStateOf(WindowPlacement.Floating to java.awt.Rectangle()) }
        s.keepPlace = { placeWas = ws.placement to window.bounds }
        s.placeBack = {
            val (pl, b) = placeWas
            if (pl == WindowPlacement.Floating) window.bounds = b else ws.placement = pl
            window.toFront()
        }
        s.maximize = { ws.placement = WindowPlacement.Maximized }
        s.raise = { window.toFront() }
        app.hide = { window.isVisible = false }
        s.fullscreen = { on ->
            val top = runCatching { Native.windowOf(window) }.getOrDefault(0L)
            if (!runCatching { Native.windowFullscreen(top, on) }.getOrDefault(false)) {
                if (on) { before = ws.placement; ws.placement = WindowPlacement.Fullscreen } else ws.placement = before
            }
        }
        DisposableEffect(Unit) {
            val resized = object : java.awt.event.ComponentAdapter() {
                override fun componentResized(e: java.awt.event.ComponentEvent) {
                    if (app.screen != Screen.Send) return
                    val top = runCatching { Native.windowOf(window) }.getOrDefault(0L)
                    if (top != 0L) s.fullFromManager(Native.windowIsFullscreen(top))
                    s.resizing()
                }
            }
            window.addComponentListener(resized)
            onDispose { window.removeComponentListener(resized) }
        }
        LizardTheme(TARGET, scrollbar = { s, m -> FadingScrollbar(s, m) }, resizeIcon = RESIZE, credits = CREDITS_HERE) {
            Screens(app.screen, app.home, app.sendUi, app.receive)
        }
        if (sendTest != null) LaunchedEffect(Unit) { sendTest.run(s) }
        if (recvTest != null) LaunchedEffect(Unit) { recvTest.run(app) }
    }
}

// ai: About's credits for what this app's packages carry besides the core (the repository's NOTICE): the runtime and the
// ai: UI's libraries; on Windows and macOS libusb and Aravis with what Aravis loads, beside the native library; on
// ai: Windows libwdi in lizard_usb.exe with the driver frameworks' co-installers
private val CREDITS_HERE = listOf(
    Credit("Kotlin", "Apache 2.0"),
    Credit("Compose Multiplatform", "Apache 2.0"),
    Credit("Skia", "BSD 3-Clause"),
    Credit("OpenJDK runtime", "GPLv2 + Classpath Exception"),
) + (if (Native.windows || Native.mac) listOf(
    Credit("libusb", "LGPLv2.1+"),
    Credit("Aravis", "LGPLv2.1+"),
    Credit("GLib", "LGPLv2.1+"),
    Credit("GNU libintl", "LGPLv2.1+"),
    Credit("libiconv", "LGPLv2.1+"),
    Credit("libxml2", "MIT"),
    Credit("libffi", "MIT"),
    Credit("PCRE2", "BSD 3-Clause"),
    Credit("zlib", "zlib"),
) else emptyList()) + (if (Native.windows) listOf(
    Credit("libwdi", "LGPLv3+"),
    Credit("Windows Driver Frameworks co-installers", "Microsoft Software License Terms"),
) else emptyList())

// ai: Every heavyweight under the window but the code canvas given a white X background (2026-11-02,
// ai: Native.windowBackground): Compose's drawing surface is one, and where the code canvas's window went (Send left) X
// ai: filled the area with the surface's own, AWT's, the desktop theme's grey (dark on a dark theme), until Skia drew it.
// ai: Their contents kept on a resize too (Native.windowKeep): AWT's ForgetGravity had X clear the whole surface at each
// ai: step of a window's resize before Skia drew it again, the whole window blinking (2026-11-02)
private fun whiten(c: java.awt.Container) {
    for (k in c.components) {
        if (k is java.awt.Canvas && k !is CodeCanvas) runCatching { Native.windowBackground(k, 0xFFFFFF); Native.windowKeep(k) }
        if (k is java.awt.Container) whiten(k)
    }
}

// ai: A scrolling column's scrollbar shown only while it is used, as an editor's overlay scrollbar: it fades in while the
// ai: column moves, while the pointer is over it and while its thumb is dragged, and out a second after the last of
// ai: those. Hidden it keeps its place, so hovering its strip brings it back.
@Composable
private fun FadingScrollbar(scroll: ScrollState, modifier: Modifier) {
    val source = remember { MutableInteractionSource() }
    val hovered by source.collectIsHoveredAsState()
    val dragged by source.collectIsDraggedAsState()
    var moving by remember { mutableStateOf(false) }
    LaunchedEffect(scroll) {
        var first = true
        snapshotFlow { scroll.value }.collectLatest {
            if (first) { first = false; return@collectLatest }
            moving = true; delay(1000); moving = false
        }
    }
    val shown = moving || hovered || dragged || scroll.isScrollInProgress
    val alpha by animateFloatAsState(if (shown) 1f else 0f, tween(if (shown) 120 else 400))
    VerticalScrollbar(rememberScrollbarAdapter(scroll), modifier.alpha(alpha), interactionSource = source)
}
