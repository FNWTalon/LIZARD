// ai: The receive session (lizard.h layer 5, 2026-10-31): a camera into a receiver, with what a receiving app does
// ai: around it, on a thread of the session's own: the phase a screen shows, the camera's settings kept per camera, a
// ai: verified file taken out of the store into dataDir/kept for the app to place (kept, placed), the camera stopped
// ai: once a file is in, one session a data folder, Save replays, the stats and the preview. Every call returns at once
// ai: (the work queued in order; a refusal is the stats' phase "error" with its why, or the log's), but close(), which
// ai: blocks until the camera is stopped, a replay written and the receiver let go: never on a UI thread. JSON field
// ai: names as core/rx/session.h says them. Desktop systems and Android: elsewhere the constructor throws.
package dev.lizard

// ai: dataDir: the session's folder (store/, kept/, replays/, receive.lock, cameras.json); assets: the GPU's kernels
// ai: and nets (null: the CPU decoder); decoder auto, gpu or cpu; layout 1:1 or 2:1;
// ai: forgetSettings: the camera settings held for this session alone (a test's run)
class Session(
  dataDir: String,
  assets: String? = null,
  cacheDir: String? = null,
  decoder: String = "auto",
  layout: String = "1:1",
  forgetSettings: Boolean = false,
) : Handle(SessionNative.create(assets, cacheDir, dataDir, decoder, layout, forgetSettings), SessionNative::free) {
  // ai: the camera by id ("" the first listed), its kept settings taken up; closeCamera stops and closes it
  fun open(camera: String) = SessionNative.open(live(), camera)
  fun closeCamera() = SessionNative.closeCamera(live())
  // ai: zoom, exposure (us), gain or any control the camera lists (automatic: the camera's own); "mode" an index
  fun set(control: String, value: Double, automatic: Boolean = false) = SessionNative.set(live(), control, value, automatic)
  // ai: mode "" the camera's kept mode, an index, or "WxH@fps"
  fun start(mode: String = "") = SessionNative.start(live(), mode)
  fun stop() = SessionNative.stop(live())
  // ai: another decoder or crop: the receiver made again, the transfer in hand going on
  fun reconfigure(decoder: String, layout: String) = SessionNative.reconfigure(live(), decoder, layout)
  fun batchCap(frames: Int) = SessionNative.batchCap(live(), frames)
  // ai: the receiver's stats with the session's (phase, why, camera, live, asked, autoRan, kept, replays); never waits
  val stats: String get() = SessionNative.stats(live())
  // ai: the camera open ({info, modes, pick, controls, mode, streaming, kept}), "{}" none; never waits
  val camera: String get() = SessionNative.camera(live())
  // ai: the newest crop newer than since, at most side px on its long side, into out: [seq, w, h], or null
  fun preview(since: Long, side: Int, out: ByteArray): LongArray? = SessionNative.preview(live(), since, side, out)
  // ai: an unfinished transfer let go by its root or id; the transfer in hand let go
  fun forget(root: String) = SessionNative.forget(live(), root)
  fun clear() = SessionNative.clear(live())
  // ai: Save replays on or off; meta the platform's fields for each run's meta.json (a JSON object)
  fun replays(on: Boolean, meta: String = "{}") = SessionNative.replays(live(), on, meta)
  // ai: the platform's own fields (a JSON object), merged into the stats and each replay row
  fun extra(json: String) = SessionNative.extra(live(), json)

  // ai: Android's hook ("" done, else why; elsewhere refused): the preview surface (an android.view.Surface, null drops
  // ai: it), waited for until the camera writes to it or no longer does (not on the UI thread)
  fun previewWindow(surface: Any?): String = SessionNative.previewWindow(live(), surface)

  companion object {
    // ai: The session's natives registered by whichever library carries them at its load: the binding's liblizard
    // ai: (Native loads it; its JNI_OnLoad registers these too), or an app's own loaded first (the desktop app's). The
    // ai: first use of Session finds which: registered already, or the binding's library loaded now. Here, not in
    // ai: SessionNative's own init: a library's JNI_OnLoad finds SessionNative to register it, which initialises it
    // ai: before its methods are registered.
    init {
      try { SessionNative.loaded() } catch (_: UnsatisfiedLinkError) { Native.version() }
    }

    // ai: every camera listed (opens devices to look: not on a UI thread), and a cheap mark of what is plugged in
    fun cameras(): String = SessionNative.cameras()
    fun camerasStamp(): String = SessionNative.camerasStamp()
    // ai: dataDir's kept/ files not yet placed, oldest first: [{id, name, type, root, path, at}]; one placed ("" or why)
    fun kept(dataDir: String): String = SessionNative.kept(dataDir)
    fun placed(dataDir: String, id: String): String = SessionNative.placed(dataDir, id)
    // ai: the unfinished transfers, newest first: [{id, name, length, fraction, root, at}]; one deleted with no session
    // ai: ("" or why: refused while a session holds the folder)
    fun partials(dataDir: String): String = SessionNative.partials(dataDir)
    fun deletePartial(dataDir: String, id: String): String = SessionNative.deletePartial(dataDir, id)
    // ai: Android's PC link, process-wide and no session's ("" done, else why; elsewhere refused): up while a cable is
    // ai: in (its hello naming layout) and down after, its frames packed or raw, its state (JSON, "{}" elsewhere); a USB
    // ai: accessory's fd (detached) to its bridge, which closes it
    fun link(up: Boolean, layout: String): String = SessionNative.link(up, layout)
    fun linkPack(on: Boolean): String = SessionNative.linkPack(on)
    fun linkState(): String = SessionNative.linkState()
    fun usbAccessory(fd: Int): String = SessionNative.usbAccessory(fd)
  }
}

// ai: The session's natives (api/jni_session.cpp), used through Session alone (its companion's init sees them there)
internal object SessionNative {
  @JvmStatic external fun loaded(): Boolean
  @JvmStatic external fun create(assets: String?, cacheDir: String?, dataDir: String, decoder: String, layout: String, forgetSettings: Boolean): Long
  @JvmStatic external fun free(h: Long)
  @JvmStatic external fun cameras(): String
  @JvmStatic external fun camerasStamp(): String
  @JvmStatic external fun open(h: Long, id: String)
  @JvmStatic external fun closeCamera(h: Long)
  @JvmStatic external fun set(h: Long, control: String, value: Double, automatic: Boolean)
  @JvmStatic external fun start(h: Long, mode: String)
  @JvmStatic external fun stop(h: Long)
  @JvmStatic external fun reconfigure(h: Long, decoder: String, layout: String)
  @JvmStatic external fun batchCap(h: Long, n: Int)
  @JvmStatic external fun stats(h: Long): String
  @JvmStatic external fun camera(h: Long): String
  @JvmStatic external fun preview(h: Long, since: Long, side: Int, out: ByteArray): LongArray?
  @JvmStatic external fun forget(h: Long, root: String)
  @JvmStatic external fun clear(h: Long)
  @JvmStatic external fun replays(h: Long, on: Boolean, meta: String)
  @JvmStatic external fun extra(h: Long, json: String)
  @JvmStatic external fun previewWindow(h: Long, surface: Any?): String
  @JvmStatic external fun link(up: Boolean, layout: String): String
  @JvmStatic external fun linkPack(on: Boolean): String
  @JvmStatic external fun linkState(): String
  @JvmStatic external fun usbAccessory(fd: Int): String
  @JvmStatic external fun kept(dataDir: String): String
  @JvmStatic external fun placed(dataDir: String, id: String): String
  @JvmStatic external fun partials(dataDir: String): String
  @JvmStatic external fun deletePartial(dataDir: String, id: String): String
}
