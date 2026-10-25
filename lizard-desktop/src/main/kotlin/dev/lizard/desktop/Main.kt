package dev.lizard.desktop

import androidx.compose.ui.window.application
import java.awt.EventQueue
import kotlin.system.exitProcess

// ai: LIZARD on a PC (2026-10-04 the sender, the web sender's page as a window; since 2026-10-25 the app, App.kt: Home,
// ai: Send and Receive). The sender's column on the left and the code area on the right, the code drawn there by the
// ai: native presenter; the receiver's column and its camera's crop beside it. LIZ_SENDER_TEST (SenderTest.kt) and
// ai: LIZ_RECEIVER_TEST (ReceiverTest.kt) run an integration check instead of waiting for a hand, at most one of them.
// ai: args: files and folders to send, added to the selection (a terminal, a file manager's Open with through the
// ai: package's desktop entry, Exec %F); an app already running takes them instead (Instance.kt), and this launch ends.
// ai: The first screen: Home; Send with files given or under LIZ_SENDER_TEST; Receive under LIZ_RECEIVER_TEST.
fun main(args: Array<String>) {
    // ai: AWT erases nothing under the presenter's pictures (the code area's canvas is the swapchain's)
    System.setProperty("sun.awt.noerasebackground", "true")
    val sendTest = SenderTest.parse(System.getenv("LIZ_SENDER_TEST"))
    val recvTest = ReceiverTest.parse(System.getenv("LIZ_RECEIVER_TEST"))
    if (sendTest != null && recvTest != null) { println("FAIL LIZ_SENDER_TEST and LIZ_RECEIVER_TEST are both set"); exitProcess(1) }
    if (sendTest != null || recvTest != null) {
        Prefs.forget()
        Thread.setDefaultUncaughtExceptionHandler { _, e ->
            val why = "${e.javaClass.simpleName}: ${e.message}"
            if (recvTest != null) ReceiverTest.fail(why) else SenderTest.fail(why)
        }
    }
    val given = args.map { java.io.File(it).absoluteFile }.filter { it.exists() }
    val testing = sendTest != null || recvTest != null
    if (!testing && given.isNotEmpty() && Instance.handOff(given)) return
    val app = App()
    val s = app.send
    try { Native.load() } catch (e: Throwable) { app.loadError = "The native library could not be loaded: ${e.message}" }
    sendTest?.apply(s)
    recvTest?.apply(app)
    app.screen = when {
        recvTest != null -> Screen.Receive
        sendTest != null || given.isNotEmpty() -> Screen.Send
        else -> Screen.Home
    }
    if (!testing) {
        Instance.listen { files -> EventQueue.invokeLater { app.handIn(files) } }
        if (given.isNotEmpty()) { if (s.test) s.chooseTest(false); s.pick(given) }
    }
    application { LizardWindow(app, sendTest, recvTest) }
}
