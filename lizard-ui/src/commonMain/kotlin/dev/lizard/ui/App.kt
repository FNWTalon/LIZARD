package dev.lizard.ui

import androidx.compose.runtime.Composable

// ai: The app's screens (both apps' since 2026-10-31): Home, Send, Receive, and a bundle's files where a platform opens a
// ai: bundle in the app (the phone's). Each app's shell keeps which shows and what entering and leaving one does (the
// ai: camera started and stopped, a send stopped), and hands in its hosts.
enum class Screen { Home, Send, Receive, Files }

@Composable
fun Screens(screen: Screen, home: HomeHost, send: SendHost, receive: ReceiveHost, files: @Composable () -> Unit = {}) = when (screen) {
    Screen.Home -> HomeScreen(home)
    Screen.Send -> SendScreen(send)
    Screen.Receive -> ReceiveScreen(receive)
    Screen.Files -> files()
}
