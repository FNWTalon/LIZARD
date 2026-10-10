package dev.lizard.receiver

import android.content.res.Configuration
import android.graphics.Matrix
import android.graphics.RectF
import android.view.Surface
import android.view.TextureView
import android.view.WindowManager
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.requiredSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.res.painterResource
import androidx.compose.runtime.Composable
import androidx.compose.runtime.movableContentOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView

// ai: Receive (2026-10-01): the bar, fixed, and under it the camera's crop, the transfer's panel, then Settings and
// ai: Developer Tools, rows that open (the camera still running, so its readouts can be watched while aiming), and
// ai: About; the words are Readout.kt's, the web receiver's. As the web page since 2026-10-02: landscape is a column on
// ai: the left, the bar over the rest scrolling on its own and a hairline on its right, and the camera on the right, as
// ai: big as the rest of the screen allows and centred there (the camera on the left before, 2026-10-01); portrait the
// ai: bar over one column that scrolls. Settings in place of the transfer's panel, by a gear in the bar, until then.
@Composable
internal fun MainActivity.ReceiveScreen() {
    val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    // ai: one preview view for both layouts: a second TextureView would destroy the first one's surface, and the
    // ai: camera with it, at every turn of the phone
    val preview = remember { movableContentOf<Modifier> { m -> Preview(m) } }
    // ai: the other tool at the bar's end, as the web's #swap (2026-10-06)
    val bar: @Composable () -> Unit = { TopBar("Receive", onBack = { go(MainActivity.Screen.Home) }) { SwapBtn("Sender") { go(MainActivity.Screen.Send) } } }
    val side: @Composable () -> Unit = {
        TransferPanel()
        Fold("Settings", isOpen("recvSettings"), { toggle("recvSettings") }) { ReceiveSettings() }
        Fold("Developer Tools", isOpen("recvAdvanced"), { toggle("recvAdvanced") }) { ReceiveAdvanced() }
        About()
        Spacer(Modifier.height(24.dp))
    }
    if (!landscape) Column(Modifier.fillMaxSize().background(Bg).safeDrawingPadding().padding(horizontal = 16.dp)) {
        bar()
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            Spacer(Modifier.height(8.dp))
            preview(Modifier.fillMaxWidth().heightIn(max = (LocalConfiguration.current.screenHeightDp * 0.55f).dp))
            side()
        }
    } else BoxWithConstraints(Modifier.fillMaxSize().background(Bg).safeDrawingPadding()) {
        // ai: the web's 20rem column, less where the screen is narrow; collapsed (its button at the bar's end, kept:
        // ai: recvCollapsed, 2026-10-02), a 56 dp rail and the camera the rest
        // ai: no narrower than Open, Share and Save in a row with no label folded (2026-10-07; two cells the day before)
        val sideLo = maxOf(CellsFloor(listOf("Open", "Share", "Save"), 3), CellsFloor(listOf("Start", "Stop"), 1))
        val sideW = SideWidth(maxWidth, sideDp, sideLo)
        val screenW = maxWidth
        Row(Modifier.fillMaxSize()) {
            // ai: Home is the rail's last square in landscape, not the column's bar (2026-10-05): expanding the panel put
            // ai: the bar's back arrow where the expand button had been, and a second tap left the receiver. Portrait
            // ai: has no rail and keeps it in its bar.
            if (isOpen("recvCollapsed")) Rail(onClick = { toggle("recvCollapsed") },
                foot = { Square { IconBtn(R.drawable.ic_back, "Home") { go(MainActivity.Screen.Home) } } }) { RailSquares() }
            else Column(Modifier.width(sideW).fillMaxHeight().padding(horizontal = 16.dp)) {
                TopBar("Receive", onBack = null) { CollapseBtn(false) { toggle("recvCollapsed") } }
                Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) { side() }
            }
            // ai: the column's edge, dragged to resize it (2026-10-05); the rail's a plain hairline
            if (isOpen("recvCollapsed")) VerticalDivider(color = Line)
            else SideEdge(sideW, screenW, sideLo, onStart = { sideDragging = true }, onDrag = { sideDp = it.value }, onDone = { sideDragging = false; saveSide() })
            BoxWithConstraints(Modifier.weight(1f).fillMaxHeight().padding(16.dp), contentAlignment = Alignment.Center) {
                val ratio = crop().ratio
                val viewW = minOf(maxHeight * ratio, maxWidth)
                preview(Modifier.width(viewW).height(viewW / ratio))
            }
        }
    }
}

// ai: The collapsed rail's squares (2026-10-02): the
// ai: camera's pause or play (toggleCamera, as Start and Stop camera), the last second's rate while a file or the test
// ai: stream is read (its figure over its unit, Readout.rate), under it the share of a file in while one comes (its
// ai: percent over "%", PctSquare; 2026-10-09), and once the file is kept a green check in the rate's
// ai: square (a mark, no action), then Open and Save, the panel's own actions, in the foreground's colour (2026-10-05;
// ai: the green tick opened the file until then). While the phone throttles (the heat warning's test, Parts.kt
// ai: heatText), its red triangle above the play or pause, or in the green check's square once the file is kept
// ai: (2026-10-08).
@Composable
private fun MainActivity.RailSquares() {
    val running = phase == Engine.Phase.On || phase == Engine.Phase.Starting || phase == Engine.Phase.Loading
    val testing = phase == Engine.Phase.On && rx.state == "test"
    val reading = phase == Engine.Phase.On && ((rx.hasFile && !rx.verified) || testing)
    // ai: the received file's squares give way to the test stream's rate while the camera reads it (2026-10-05)
    val kept = if (rx.verified && !testing) files.firstOrNull { it.root == root } else null
    val hot = heatText(heat, clocks, engine.cameraSlow)
    if (hot != null && kept == null) HeatSquare(hot)
    Square {
        IconBtn(if (running) R.drawable.ic_pause else R.drawable.ic_play, if (running) "Stop camera" else "Start camera",
            enabled = granted && phase != Engine.Phase.Starting) { toggleCamera() }
    }
    if (kept == null) {
        RateSquare(if (reading) rx.goodputKBs else null)
        if (reading && !testing) PctSquare(Readout.fraction(rx))
    } else {
        if (hot != null) HeatSquare(hot)
        else Square { Icon(painterResource(R.drawable.ic_check), contentDescription = "Received ${kept.name}", Modifier.size(24.dp), tint = Good) }
        Square { IconBtn(R.drawable.ic_open, "Open ${kept.name}") { open(kept) } }
        Square { IconBtn(R.drawable.ic_save, "Save ${kept.name}") { saveCopy(kept) } }
    }
}

// ai: The transfer: the state, the meter, the figures (progress, size, time left), then the actions: once the file is
// ai: kept, Open (the solid one), Share and Save on one row, and Start or Stop (the camera's) alone under them, the
// ai: full width (2026-10-07; two rows of two for a day before that; "Start
// ai: camera" and "Stop camera" until 2026-10-06, cut in a cell; 2026-10-05: "Save a copy" renamed, and Receive again
// ai: deleted, a second Start camera). A label never folds or shortens (Parts.kt Btn): the column's floor is what
// ai: three cells of the widest of those labels need (CellsFloor). Without the camera's permission, why and the one button that
// ai: gets it. Its first-run tip went 2026-10-02, as the web receiver's.
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun MainActivity.TransferPanel() {
    val on = phase == Engine.Phase.On
    val running = on || phase == Engine.Phase.Starting || phase == Engine.Phase.Loading
    val ln = if (!granted) Readout.Line(if (denied) "The camera is off for LIZARD. Allow it in Android's settings to receive." else "LIZARD needs the camera to read the code on the other screen.")
        else Readout.line(on, phase == Engine.Phase.Starting, phase == Engine.Phase.Loading, (phase as? Engine.Phase.Error)?.why, rx)
    // ai: Show statistics off (Developer Tools, 2026-10-10): no green line once a file is in, the meter and buttons as before
    val shown = ln.text.isNotEmpty() && !(ln.tone == Readout.Tone.Good && settings.stats == "off")
    if (shown) Text(ln.text, style = MaterialTheme.typography.titleMedium, color = toneColour(ln.tone), modifier = Modifier.padding(top = 12.dp, bottom = 2.dp))
    Meter(ln.frac)
    if (ln.nums.isNotEmpty()) Text(ln.nums, style = MaterialTheme.typography.bodyMedium, color = Muted)
    HeatWarning(heat, clocks, engine.cameraSlow, Modifier.padding(top = 8.dp))
    // ai: the file's buttons give way while the camera reads the test stream (2026-10-05), as the state line does
    val kept = if (rx.verified && !(on && rx.state == "test")) files.firstOrNull { it.root == root } else null
    Column(Modifier.fillMaxWidth().padding(vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (!granted) Btn(if (denied) "Open settings" else "Allow camera", Kind.Primary, modifier = Modifier.fillMaxWidth()) { if (denied) openAppSettings() else askCamera() }
        else {
            // ai: one solid button at a time: Start while nothing waits, else Open
            val camera: @Composable (Modifier) -> Unit = { m ->
                Btn(if (running) "Stop" else "Start", if (!running && kept == null) Kind.Primary else Kind.Tonal, enabled = phase != Engine.Phase.Starting, modifier = m) { toggleCamera() }
            }
            if (kept != null) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Btn("Open", Kind.Primary, modifier = Modifier.weight(1f)) { open(kept) }
                Btn("Share", modifier = Modifier.weight(1f)) { share(kept) }
                Btn("Save", modifier = Modifier.weight(1f)) { saveCopy(kept) }
            }
            camera(Modifier.fillMaxWidth())
        }
    }
    // ai: the rate itself is the collapsed rail's and Developer Tools' lab line's (2026-10-05; a line under the buttons
    // ai: from earlier that day)
    if (note.isNotEmpty()) Text(note, style = MaterialTheme.typography.bodyMedium, color = if (note.startsWith("Saved")) Muted else Bad)
}

// ai: The preview upright on a turned display (Camera2 with a TextureView: the camera turns its buffers for the
// ai: display's natural orientation only; Google's Camera2 sample's configureTransform): at 90 or 270 degrees the
// ai: view maps onto the buffer's turned rectangle, scaled back to the view and rotated against the display.
private fun MainActivity.turn(v: TextureView, pw: Float, ph: Float) {
    val vw = v.width.toFloat()
    val vh = v.height.toFloat()
    if (vw == 0f || vh == 0f) return
    @Suppress("DEPRECATION")
    val rot = (getSystemService(android.content.Context.WINDOW_SERVICE) as WindowManager).defaultDisplay.rotation
    val m = Matrix()
    val view = RectF(0f, 0f, vw, vh)
    val cx = view.centerX()
    val cy = view.centerY()
    if (rot == Surface.ROTATION_90 || rot == Surface.ROTATION_270) {
        val buf = RectF(0f, 0f, ph, pw)
        buf.offset(cx - buf.centerX(), cy - buf.centerY())
        m.setRectToRect(view, buf, Matrix.ScaleToFit.FILL)
        val k = maxOf(vh / ph, vw / pw)
        m.postScale(k, k, cx, cy)
        m.postRotate(90f * (rot - 2), cx, cy)
    } else if (rot == Surface.ROTATION_180) m.postRotate(180f, cx, cy)
    v.setTransform(m)
}

// ai: the crop the decoder reads, as the screen shows it: 1:1 the frame's centre square, 2:1 its centre region twice
// ai: as long as high along the long side (rx/receiver.cpp push). The sensor is landscape and the camera turns the
// ai: preview upright, so a portrait screen shows the long side vertically. fullX / cropX is how much wider than
// ai: the crop the whole preview is, and the same for the height; ratio is the crop's width over its height.
internal class Crop(val fullX: Float, val fullY: Float, val cropX: Float, val cropY: Float) { val ratio get() = cropX / cropY }

@Composable
internal fun MainActivity.crop(): Crop {
    val c = cam
    val portrait = LocalConfiguration.current.orientation == Configuration.ORIENTATION_PORTRAIT
    val w = (c?.size?.width ?: 16).toFloat()
    val h = (c?.size?.height ?: 9).toFloat()
    val lo = minOf(w, h)
    val hi = maxOf(w, h)
    val two = settings.layout == "2:1"
    val side = if (two) minOf(lo, hi / 2) else lo
    val cropLong = if (two) 2 * side else side
    return Crop(if (portrait) lo else hi, if (portrait) hi else lo, if (portrait) side else cropLong, if (portrait) cropLong else side)
}

// ai: Only the crop is shown, as the web receiver shows its crop: the whole preview is laid out at the size that
// ai: puts the crop over the box, and clipped. The box's size is the caller's (portrait: the width, up to 55% of
// ai: the screen's height; landscape: the height), its corners square as the web's. Over it, as the web's: "Camera off"
// ai: (or why there is none) with its icon, and "Getting ready" with a spinner until the receiver first answers.
@Composable
internal fun MainActivity.Preview(modifier: Modifier) {
    val c = cam
    val g = crop()
    // ai: square corners (2026-10-02; 12 dp rounded before); the crop still
    // ai: clipped to its box, the preview being laid out larger
    Box(modifier.clipToBounds().background(Soft), contentAlignment = Alignment.Center) {
        BoxWithConstraints(Modifier.aspectRatio(g.ratio, matchHeightConstraintsFirst = true).clipToBounds(),
            contentAlignment = Alignment.Center) {
            val pw = (c?.preview?.width ?: 16).toFloat()
            val ph = (c?.preview?.height ?: 9).toFloat()
            // ai: the layout listener turns by the preview's current shape, not the one the view was made with
            // ai: (2026-10-07): the factory runs once, and a preview of another shape (a lens with no size of the
            // ai: capture's shape) would be turned by the old one at every layout, stretched
            val shape = rememberUpdatedState(pw to ph)
            AndroidView({ ctx ->
                TextureView(ctx).apply {
                    surfaceTextureListener = engine.textureListener
                    addOnLayoutChangeListener { v, _, _, _, _, _, _, _, _ -> val (w, hh) = shape.value; turn(v as TextureView, w, hh) }
                }
            }, Modifier.requiredSize(maxWidth * (g.fullX / g.cropX), maxHeight * (g.fullY / g.cropY)), update = { turn(it, pw, ph) })
            // ai: the 2:1 crop's centre guide (2026-09-30): a red 1 dp line across the middle of the crop's long side,
            // ai: where the receiver cuts the two halves (rx/receiver.cpp cropsOf), whenever the crop is 2:1. Only a
            // ai: mark on the preview: the halves are handed to the decoder as plain frames.
            if (settings.layout == "2:1") Canvas(Modifier.matchParentSize()) {
                val wide = g.cropX >= g.cropY
                val a = if (wide) Offset(size.width / 2, 0f) else Offset(0f, size.height / 2)
                val b = if (wide) Offset(size.width / 2, size.height) else Offset(size.width, size.height / 2)
                drawLine(Color(0xFFE53935), a, b, strokeWidth = 1.dp.toPx())
            }
            val over = when {
                !granted -> "No camera yet"
                phase == Engine.Phase.Idle || phase is Engine.Phase.Error -> "Camera off"
                else -> ""
            }
            val waiting = granted && Readout.gettingReady(phase == Engine.Phase.On, phase == Engine.Phase.Starting, phase == Engine.Phase.Loading, rx)
            if (over.isNotEmpty() || waiting) Column(Modifier.matchParentSize().background(Soft), verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
                horizontalAlignment = Alignment.CenterHorizontally) {
                if (over.isNotEmpty()) Icon(painterResource(R.drawable.ic_camera_off), contentDescription = null, Modifier.size(24.dp), tint = Muted)
                else CircularProgressIndicator(Modifier.size(28.dp), color = Fg, trackColor = Line, strokeWidth = 3.dp)
                Text(over.ifEmpty { "Getting ready" }, style = MaterialTheme.typography.bodyMedium, color = Muted, textAlign = TextAlign.Center)
            }
        }
    }
}
