package dev.lizard.ui

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.unit.dp

// ai: The icons: Feather's (MIT) on a 24 grid, 2 px strokes, round caps and joins, as the web pages draw them
// ai: (lizard-web/ui.mjs ICONS); and LIZARD's mark (lizard-web/icon.svg: the corner marks, the dark ring between them,
// ai: a grey picture, on white), the bar's and the desktop window's. One set for both apps (2026-10-31; the Android app's
// ai: vector drawables and the desktop's Parts.kt before). The Android launcher's icon and its shortcuts' stay
// ai: resources (res/drawable), which the launcher reads.
object Icons {
    private fun stroked(name: String, vararg d: String) = ImageVector.Builder(name, 24.dp, 24.dp, 24f, 24f).apply {
        for (p in d) addPath(addPathNodes(p), stroke = SolidColor(Fg), strokeLineWidth = 2f, strokeLineCap = StrokeCap.Round, strokeLineJoin = StrokeJoin.Round)
    }.build()
    // ai: a circle as a path (Feather draws some of its icons with <circle>)
    private fun circle(cx: Float, cy: Float, r: Float) = "M${cx - r} ${cy}a$r $r 0 1 0 ${2 * r} 0a$r $r 0 1 0 ${-2 * r} 0"
    val chevron = stroked("chevron", "M6,9l6,6 6,-6")
    val collapse = stroked("collapse", "M11,17l-5,-5 5,-5M18,17l-5,-5 5,-5")
    val play = stroked("play", "M6,3l14,9 -14,9z")
    val pause = stroked("pause", "M6,4h4v16h-4z M14,4h4v16h-4z")
    val file = stroked("file", "M14,2H6a2,2 0,0 0,-2 2v16a2,2 0,0 0,2 2h12a2,2 0,0 0,2 -2V8z", "M14,2v6h6")
    val folder = stroked("folder", "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z")
    val image = stroked("image", "M5,3h14a2,2 0,0 1,2 2v14a2,2 0,0 1,-2 2H5a2,2 0,0 1,-2 -2V5a2,2 0,0 1,2 -2z", circle(8.5f, 8.5f, 1.5f), "M21,15L16,10L5,21")
    val swap = stroked("swap", "M8 3L4 7l4 4M4 7h16M16 21l4-4-4-4M20 17H4")
    val camera = stroked("camera", "M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z", circle(12f, 13f, 4f))
    val cameraOff = stroked("camera-off", "M1 1l22 22M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34m-7.72-2.06a4 4 0 1 1-5.56-5.56")
    val monitor = stroked("monitor", "M4 3h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z", "M8 21h8M12 17v4")
    val check = stroked("check", "M22 11.08V12a10 10 0 1 1-5.93-9.14", "M22 4L12 14.01l-3-3")
    val open = stroked("open", "M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6", "M15 3h6v6", "M10 14L21 3")
    val save = stroked("save", "M21,15v4a2,2 0 0,1 -2,2H5a2,2 0 0,1 -2,-2v-4", "M7,10l5,5 5,-5", "M12,15V3")
    val more = stroked("more", circle(12f, 5f, 1f), circle(12f, 12f, 1f), circle(12f, 19f, 1f))
    val alert = stroked("alert", "M10.29,3.86L1.82,18a2,2 0,0 0,1.71 3h16.94a2,2 0,0 0,1.71 -3L13.71,3.86a2,2 0,0 0,-3.42 0z", "M12,9L12,13", "M12,17L12.01,17")
    val back = stroked("back", "M19 12H5M12 19l-7-7 7-7")
    val x = stroked("x", "M18,6 L6,18 M6,6 L18,18")
    val mark = ImageVector.Builder("mark", 64.dp, 64.dp, 64f, 64f).apply {
        addPath(addPathNodes("M0 0h64v64h-64z"), fill = SolidColor(Color.White))
        // ai: the ink spans 70% of the square (2026-10-10, the standard lizard-web/icon.svg states)
        addGroup(pivotX = 32f, pivotY = 32f, scaleX = 1.24444f, scaleY = 1.24444f)
        addPath(addPathNodes("M14 14h9v9h-9zM41 14h9v9h-9zM14 41h9v9h-9zM41 41h9v9h-9zM25 16h14v3H25zM25 45h14v3H25zM16 25h3v14h-3zM45 25h3v14h-3z"), fill = SolidColor(Fg))
        addPath(addPathNodes("M25 25h7v7h-7zM32 32h7v7h-7z"), fill = SolidColor(Color(0xFF777777)))
        addPath(addPathNodes("M32 25h7v7h-7zM25 32h7v7h-7z"), fill = SolidColor(Color(0xFFBBBBBB)))
        clearGroup()
    }.build()
}
