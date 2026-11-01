// ai: The receive session (2026-10-31; the desktop's RxSession of 2026-10-25 grown into the engine): one camera
// ai: (cam/camera.h) feeding one Receiver (receiver.h), and around them what every app's receive screen needs alike:
// ai: the phase it shows, the per-camera settings kept, the rules a poll used to apply (a verified file taken out of the
// ai: store into kept/, the camera stopped once a file is in, a stream that ended closed), the one-session lock on the
// ai: data folder, Save replays, the stats, the preview; and for a camera that can delay a capture (Android's, cam/
// ai: android.cpp) the phase lock (phase_host.h over phase_lock.h) fed from its capture results. The C API's layer 5 (lizard.h liz_session_*)
// ai: is its face, and through it the Kotlin binding's dev.lizard.Session (the desktop app and, since 2026-10-31, the
// ai: Android app); lizard_receive and cam_check use it directly.
// ai: The data folder (Config::dataDir): store/ (the transfer's: store/lizard-xfer, kept across sessions to resume),
// ai: kept/<id>/ (a verified file taken out of the store: filing.json {name, type, root} and file/<name>; id is the
// ai: root's first 16 hex, a dash and a count), replays/<run>.lizard, receive.lock, cameras.json. A take that fails
// ai: (the disk full, the file held by another program) leaves the file in the store, which keeps it (store.h
// ai: fileStore), and is tried again at each start, a camera's close, the session's first moments and its end; a take
// ai: whose copy is whole and whose source would not go is a take.
// ai: Threads: one worker thread of the session's runs every command in the order asked and, between them, the rules
// ai: every 100 ms; a command returns at once with a future of its refusal ("" none). stats, camera, preview, series
// ai: and streaming wait on no command. The camera's frames go to the receiver on the camera's capture thread.
// ai: Phases: Idle; Starting from a start until its stream runs (the lock, the camera, the receiver made); Loading while
// ai: the stream runs and the receiver is not yet past "starting" (its first plan), and through a reconfigure; On; Error
// ai: with its sentence. start, stop and reconfigure set theirs at the call, so a stats read right after shows it; an
// ai: outcome of a command overtaken by a later start or stop changes no phase.
// ai: cameras.json: {"version": 1, "cameras": {"<id>": {"mode": {"w", "h", "fps", "format"}, "zoom": {"1:1": v, "2:1":
// ai: v}, "exposure": us or null, "gain": v or null}}}: a camera's mode, its zoom a crop at a time, its exposure and
// ai: gain (null or absent: the camera's own, automatic where it can be); written whole beside it and renamed over it.
#pragma once
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace lizard {

namespace cam { class Camera; }

// ai: A camera as Session::camera() describes it, from the camera itself (lizard_receive list's line): {"info",
// ai: "modes", "pick", "controls"}; its controls and modes read off the camera, a camera call.
std::string describeCamera(cam::Camera& c);

class Session {
 public:
  struct Config {
    std::string assets;      // ai: the GPU tree (ReceiverConfig.assets)
    std::string cacheDir;    // ai: the pipeline cache's folder; "" none
    std::string dataDir;     // ai: the store, kept/, replays/, the lock, the camera settings (the header)
    std::string decoder = "auto", layout = "1:1";   // ai: auto | gpu | cpu (and pc on Android: stream.h); 1:1 | 2:1
    // ai: the camera settings held for this session alone, none read from or written to cameras.json (a test's run)
    bool forgetSettings = false;
    std::function<void(const std::string&)> log;
  };
  // ai: At once: the worker started, which takes the data folder's lock (waiting up to 10 s for a session of this
  // ai: process on the same folder that is closing), adopts the replays there and takes out any finished file an
  // ai: earlier session left in the store. Null with why for a decoder or a
  // ai: layout it does not know, or no data folder. The receiver is made at the first start (prepare: before it).
  static std::unique_ptr<Session> create(const Config& c, std::string& why);
  // ai: Blocks: the commands asked before it run, then the camera stopped and closed (its controls put back), a
  // ai: recording replay ended and written, the receiver let go, the lock released. Not on a UI thread.
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  using Done = std::shared_future<std::string>;
  // ai: The receiver made now where none is (a start makes it otherwise), with the decoder and layout asked last.
  Done prepare();
  // ai: The camera by id (the one open before stopped and closed; the same id open already: nothing). "" is the first
  // ai: listed; an id that does not open and is not listed (a camera unplugged since it was chosen) the first listed in
  // ai: its place (said in the log), but a phone typed by its address (phone:<ipv4>:<port>) or an upload (upload:<path>),
  // ai: which no list holds. The cameras are listed only then (a listing probes devices and listens for phones). Its
  // ai: kept settings are taken up (zoom for the crop, exposure, gain: set after each start). A refusal is the phase's
  // ai: Error ("No camera was found." where none is listed).
  Done open(const std::string& id);
  // ai: The camera stopped and closed; the settings it was given forgotten (its kept ones stay kept); the takes tried
  // ai: again.
  Done close();
  // ai: A user's setting, kept for the camera (cameras.json) and set after each start, since every backend puts back
  // ai: at stop what set changed: zoom (its crop's), exposure (us), gain, and any control the camera lists (kept for
  // ai: this open alone); automatic for the camera's own. While a stream runs it is set at once (refused with the
  // ai: camera's word); with none it is kept alone, the camera untouched, and set at the next start. "mode": value an
  // ai: index into the camera's modes, kept as its mode, the stream restarted in it where one runs. Asks of one control
  // ai: queued and not yet run are one: the newest.
  Done set(const std::string& control, double value, bool automatic);
  // ai: The camera open streamed into the receiver (the receiver made first where none is; the lock taken where it
  // ai: was not): mode "" the camera's kept mode (else camera.h pick's for 1920 x 1080 at 60), an index, or
  // ai: "WxH@fps" (the first such, uncompressed first); one named is kept as the camera's. A camera that names its own
  // ai: crop (a phone's, an upload's: its detail "2:1, ...") makes the receiver's crop its own. The takes tried again
  // ai: first. Phase Starting at the call, Loading once the stream runs; a refusal closes the camera and is the phase's
  // ai: Error.
  Done start(const std::string& mode = "");
  // ai: The stream stopped (its thread joined), the receiver told (cameraClosed). Phase Idle at the call.
  Done stop();
  // ai: A new receiver for another decoder or crop, made on the old one's transfer (receiver.h create's from):
  // ai: a transfer under way goes on, a file finished stays; the camera stopped around the swap and started again in
  // ai: its mode, the batch cap and a recording replay handed on. With no receiver yet, what the first is made with.
  // ai: Phase Loading at the call where the camera was asked to run. A refusal: the camera closed, the phase's Error.
  Done reconfigure(const std::string& decoder, const std::string& layout);
  Done batchCap(int n);   // ai: receiver.h batchCap, kept for a receiver made later
  // ai: An unfinished transfer let go by its root or its folder's id (receiver.h forget; no receiver: its folder).
  Done forget(const std::string& root);
  // ai: The transfer in hand and its file let go (receiver.h clear), so the same file in the light is received anew
  // ai: (another transfer's finished file not yet taken out stays).
  Done clear();
  // ai: Save replays on or off: a run recorded while on and the camera On (an upload's never: it is a replay), ended
  // ai: when either stops; meta, a JSON object of the platform's own fields for the run's meta.json (the app's build).
  // ai: A run that ends whole lets the runs before it go.
  Done replays(bool on, const std::string& meta);
  // ai: The platform's own fields (a JSON object: heat, power), merged into the stats, each replay row and a run's meta
  // ai: (its end's) from now on.
  void extra(const std::string& json);

  // ai: The platform's hooks (2026-10-31, Android's; elsewhere refused with why, "" done):
  // ai: previewWindow: the app's preview surface (an ANativeWindow*, acquired here as the command runs, so the caller
  // ai: holds its own reference until the outcome; null drops it), handed to every camera opened from now on as its
  // ai: second stream (camera.h previewTo), done once the camera writes to it or no longer does (the app releases a
  // ai: surface it dropped after that). Size it to the started mode's preview
  // ai: (camera()'s modes, "preview") before the start.
  Done previewWindow(void* window);
  // ai: The PC link (stream.h; the "pc" decoder serves its frames to a PC there), process-wide and no session's: one
  // ai: link a phone, up whether a session lives or not (a PC sees the phone from any screen): up while a cable is in,
  // ai: its hello naming layout (link), down after; the frames packed as .lizard frames or raw (linkPack); its state
  // ai: (linkState: stream.h linkState's JSON with "accessory", a USB accessory's bridge up; {} elsewhere); a USB
  // ai: accessory's fd handed to its bridge (usbAccessory: false where one is up already, the fd closed; elsewhere
  // ai: false, the fd the caller's). At once; logged to the system's log.
  static bool link(bool up, const std::string& layout, std::string& why);
  static bool linkPack(bool on, std::string& why);
  static std::string linkState();
  static bool usbAccessory(int fd, std::string& why);

  // ai: The receiver's stats (receiver.h stats) with the session's beside them:
  // ai:   phase "idle" | "starting" | "loading" | "on" | "error", why (the error's sentence, "" otherwise);
  // ai:   camera {id, name, backend, detail, streaming, starts, mode {w, h, format, fps} (the started, null none), fps
  // ai:   (measured over the last second; 0 before a stream's fourth frame and with none), agreedFps, frames, dropped,
  // ai:   corrupt, convertMs, asks, askOn (camera.h Stats), ended (why a stream stopped by itself)}, null with none open;
  // ai:   live: a stream runs, so the receiver's window figures are its own (else they read 0 and test false:
  // ai:   capturedFps, processedFps, dropped, foundShare, side, heldMs, heldMaxMs, blocks, windowSecs, goodputKBs,
  // ai:   gpuMs, cpuMs, repeatShare);
  // ai:   asked {decoder, layout} (what the next receiver is made with), autoRan "GPU" | "CPU" (what auto runs
  // ai:   on; "" while the receiver in hand was not made with auto, or none is);
  // ai:   kept {taken (files this session took into kept/), root (the last one's), error (why the last take failed, ""
  // ai:   none)};
  // ai:   replays [{run, path, state "recording" | "saving" | "ready" | "failed", frames, bytes, error}], on (the switch);
  // ai:   lock: the phase lock's state (phase_lock.h State: arm, state, a, b, k, se, n, gain, pace, stood, edge, flatMs,
  // ai:   delays, refreshMs, multiple, frameMs, landMs, row; NaN as null; {} before its first decision), or {"off": why}
  // ai:   where the camera started last cannot delay a capture (camera.h delayBar); a replay's rows and meta carry the same
  // ai:   under "phase" (the Android app's key); camera.report: the camera's own (camera.h report);
  // ai:   and extra's fields. With no receiver yet: the session's fields alone.
  std::string stats();
  // ai: The camera open: {"info", "modes", "pick", "controls" (values as last read), "mode" (the index started last, -1
  // ai: none), "streaming", "kept": {"mode" (the index start("") takes, -1 none), "zoom" (its crop's), "exposure",
  // ai: "gain" (null: the camera's own)}}; {} with none. Never waits.
  std::string camera();
  // ai: The newest frame's crop as the decoder reads it (the centre square, or the 2:1 region whole), area-averaged to
  // ai: at most `side` pixels on its long side (never enlarged), with that frame's count in seq; false where no frame
  // ai: newer than `since` came. The capture thread copies the crop only while a preview was asked within the last
  // ai: second, so a first call after a pause answers false; the scaling is the caller's thread's.
  bool preview(uint64_t since, int side, std::vector<uint8_t>& out, uint32_t& w, uint32_t& h, uint64_t& seq);
  std::vector<double> series(double sinceMs);   // ai: receiver.h series ({0} with no receiver)
  bool streaming() const;

  // ai: Every backend's cameras (camera.h list): {"cameras": [{id, name, backend, detail}], "notes": [...]}; opens
  // ai: devices to look (seconds on some systems). camerasStamp: camera.h listStamp, nothing opened.
  static std::string cameras();
  static std::string camerasStamp();
  // ai: The takes in dataDir's kept/ whose file is whole, oldest first: [{id, name, type, root, path, at (ms)}].
  static std::string kept(const std::string& dataDir);
  // ai: A take placed by the app (or let go with its entry): its folder deleted. False with why.
  static bool placed(const std::string& dataDir, const std::string& id, std::string& why);
  // ai: The unfinished transfers in dataDir's store (a folder with a header and a readable meta.json), newest first:
  // ai: [{id, name, length, fraction, root, at (ms)}].
  static std::string partials(const std::string& dataDir);
  // ai: One deleted with no session: refused while a session holds the data folder (one receiving, of this process or
  // ai: another, at once), but one of this process's that is closing (let go a moment ago, its end on its own thread)
  // ai: is waited for, up to 10 s.
  static bool deletePartial(const std::string& dataDir, const std::string& id, std::string& why);

  struct Impl;

 private:
  explicit Session(std::unique_ptr<Impl> p);
  std::unique_ptr<Impl> p_;
};

// ai: a command's outcome waited for: true where it was done, else false with its refusal
inline bool done(const Session::Done& d, std::string& why) {
  why = d.get();
  return why.empty();
}

}  // namespace lizard
