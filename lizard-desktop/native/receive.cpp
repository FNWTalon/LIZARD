// ai: lizard_receive (2026-10-25): the PC receiver headless, as lizard_present is the sender: a camera (camera/
// ai: camera.h) into a receiver (rx_session.h), a line a second, a received file moved out; on every system.
// ai:   lizard_receive list
// ai:     every camera every backend lists, a JSON line each ({info, modes, pick, controls, trigger}, the camera opened
// ai:     to read them; {info, error} where it would not open), then {"notes": [...]}: why a backend saw none
// ai:   lizard_receive [--camera ID] [--mode INDEX|WxH[@FPS]] [--layout 1:1|2:1] [--decoder auto|gpu|cpu]
// ai:                  [--precision auto|int8|f16|f32] [--assets DIR] [--store DIR] [--out DIR] [--secs S]
// ai:                  [--trigger MS] [--set CONTROL=VALUE|auto]... [--json]
// ai: The camera: the one named, else the simulated one (sim:0) where LIZ_CAMERA_SIM is set (nothing listed), else the
// ai: first listed (list and this probe every /dev/video* unless LIZ_V4L2_NODES names the nodes: set and empty, none).
// ai: Its mode: by index, or the first of that size and rate (a rate within half a frame a second; none given, any),
// ai: else the nearest (camera.h pick for that size and rate); no --mode, pick's for 1920 x 1080 at 60. --set after the
// ai: start (the camera rules are applied at start, so a value asked here wins; the session keeps it, rx_session.h
// ai: set). --trigger: the camera's software trigger every MS ms, a free schedule (no phase lock: the app's Lock.kt
// ai: steers one). The store: a fresh folder in the system's temp folder, removed at the end (kept where a verified
// ai: file could not be moved out: the FAIL line says where the file is), unless --store names one (kept: a transfer
// ai: cut short resumes there). --out is made before the camera starts (a path that cannot be a folder fails then).
// ai: The GPU tree: --assets, else the build's liblizard/out. A line a
// ai: second on stdout, "rx: <state>, camera F fps, processed P fps, found S%, B blocks, G KB/s[, file X% <name>]" (the
// ai: receiver's last second of this stream, 0 until its first has closed), or with --json the whole stats JSON
// ai: (rx_session.h stats); the receiver's log on stderr.
// ai: The end: a file verified is moved into --out (the working folder by default) under its own name ("name (1).ext"
// ai: where that is taken), the last stats (a line, or the JSON with --json), "summary: T s, the file N B in S s (R
// ai: KB/s), found F% of K frames, B blocks verified, D frames decoded" (T since the camera started, S and R on the
// ai: transfer's own clock, F the share of the run's frames a symbol was found in, a 2:1 half a frame) and "file
// ai: <path>" printed, exit 0. --secs passed, SIGINT or SIGTERM with no file: the summary without the file, exit 0
// ai: where any block was verified, else 1. A refusal anywhere: "FAIL <why>" on stdout, exit 1. A usage error: exit 2.
// ai: Every argument is taken as UTF-8 (paths through fs::u8path); on Windows they come in as UTF-16 (wmain) and are
// ai: made UTF-8 first, since main's would be in the ANSI code page, which a path outside it does not survive. What it
// ai: prints is UTF-8 too.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "camera.h"
#include "json.hpp"
#include "rx_session.h"

#ifndef LIZ_ASSETS
#define LIZ_ASSETS "liblizard/out"
#endif

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

std::atomic<bool> gQuit{false};

struct Args {
  bool list = false, json = false;
  std::string camera, mode, layout = "1:1", decoder = "auto", precision = "auto", assets = LIZ_ASSETS, store, out = ".";
  double secs = 0, triggerMs = 0;
  std::vector<std::string> sets;
};

[[noreturn]] void usage(const std::string& why) {
  std::fprintf(stderr,
               "%s%susage: lizard_receive list\n"
               "       lizard_receive [--camera ID] [--mode INDEX|WxH[@FPS]] [--layout 1:1|2:1] [--decoder auto|gpu|cpu]\n"
               "                      [--precision auto|int8|f16|f32] [--assets DIR] [--store DIR] [--out DIR] [--secs S]\n"
               "                      [--trigger MS] [--set CONTROL=VALUE|auto]... [--json]\n",
               why.c_str(), why.empty() ? "" : "\n");
  std::exit(2);
}

Args parse(int argc, char** argv) {
  Args a;
  if (argc >= 2 && std::string(argv[1]) == "list") {
    if (argc > 2) usage("list takes no arguments");
    a.list = true;
    return a;
  }
  for (int i = 1; i < argc; i++) {
    const std::string k = argv[i];
    if (k == "-h" || k == "--help") usage("");
    if (k == "--json") {
      a.json = true;
      continue;
    }
    if (i + 1 >= argc) usage("no value for " + k);
    const std::string v = argv[++i];
    try {
      if (k == "--camera") a.camera = v;
      else if (k == "--mode") a.mode = v;
      else if (k == "--layout") a.layout = v;
      else if (k == "--decoder") a.decoder = v;
      else if (k == "--precision") a.precision = v;
      else if (k == "--assets") a.assets = v;
      else if (k == "--store") a.store = v;
      else if (k == "--out") a.out = v;
      else if (k == "--secs") a.secs = std::stod(v);
      else if (k == "--trigger") a.triggerMs = std::stod(v);
      else if (k == "--set") a.sets.push_back(v);
      else usage("unknown " + k);
    } catch (const std::logic_error&) { usage("not a number: " + k + " " + v); }
  }
  if (a.layout != "1:1" && a.layout != "2:1") usage("--layout is 1:1 or 2:1");
  if (a.decoder != "auto" && a.decoder != "gpu" && a.decoder != "cpu") usage("--decoder is auto, gpu or cpu");
  if (a.precision != "auto" && a.precision != "int8" && a.precision != "f16" && a.precision != "f32") usage("--precision is auto, int8, f16 or f32");
  if (a.secs < 0) usage("--secs is 0 (no end) or more");
  if (a.triggerMs < 0 || (a.triggerMs > 0 && a.triggerMs < 1)) usage("--trigger is a period of 1 ms or more");
  for (const auto& s : a.sets)
    if (s.find('=') == std::string::npos || s.front() == '=') usage("--set is CONTROL=VALUE or CONTROL=auto: " + s);
  return a;
}

double secsSince(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }

// ai: a number field of a stats object, 0 where it is missing or not a number
double num(const json& j, const char* k) {
  if (!j.is_object()) return 0;
  const auto it = j.find(k);
  return it != j.end() && it->is_number() ? it->get<double>() : 0;
}
std::string text(const json& j, const char* k) {
  if (!j.is_object()) return "";
  const auto it = j.find(k);
  return it != j.end() && it->is_string() ? it->get<std::string>() : "";
}

json parsed(const std::string& s) {
  try {
    return json::parse(s);
  } catch (const std::exception&) {
    return json::object();
  }
}

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// ai: the mode --mode names (rx_session.h camera's "modes"): an index, or WxH[@FPS]; -1 with why
int modeOf(const json& modes, const std::string& want, int pickIndex, std::string& why) {
  const int n = modes.is_array() ? int(modes.size()) : 0;
  if (!n) {
    why = "the camera lists no mode";
    return -1;
  }
  if (want.empty()) return pickIndex >= 0 && pickIndex < n ? pickIndex : 0;
  if (want.find_first_not_of("0123456789") == std::string::npos) {
    const int i = want.size() > 6 ? n : std::stoi(want);
    if (i >= n) {
      why = "the camera has no mode " + want + " (" + std::to_string(n) + " listed)";
      return -1;
    }
    return i;
  }
  unsigned w = 0, h = 0;
  double fps = 0;
  char x = 0;
  int got = std::sscanf(want.c_str(), "%u%c%u@%lf", &w, &x, &h, &fps);
  if (got < 3 || (x != 'x' && x != 'X') || !w || !h) {
    why = "--mode is an index or WxH[@FPS]: " + want;
    return -1;
  }
  if (got < 4) fps = 0;
  for (int i = 0; i < n; i++) {
    const json& m = modes[size_t(i)];
    if (num(m, "w") == w && num(m, "h") == h && (fps <= 0 || std::fabs(num(m, "fps") - fps) < 0.5)) return i;
  }
  std::vector<lizard::cam::Mode> ms(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) {
    ms[size_t(i)].width = uint32_t(num(modes[size_t(i)], "w"));
    ms[size_t(i)].height = uint32_t(num(modes[size_t(i)], "h"));
    ms[size_t(i)].fps = num(modes[size_t(i)], "fps");
    ms[size_t(i)].fpsMin = num(modes[size_t(i)], "fpsMin");
    ms[size_t(i)].compressed = modes[size_t(i)].value("compressed", false);
  }
  const int i = lizard::cam::pick(ms, w, h, fps);
  std::fprintf(stderr, "lizard_receive: no mode %s, the nearest is %d\n", want.c_str(), i);
  return i >= 0 ? i : 0;
}

// ai: name in dir, or "name (k).ext" with the first k not taken
fs::path freeName(const fs::path& dir, const fs::path& name) {
  fs::path p = dir / name;
  std::error_code e;
  const std::string stem = name.stem().u8string(), ext = name.extension().u8string();
  for (int k = 1; fs::exists(p, e) && k < 10000; k++) p = dir / fs::u8path(stem + " (" + std::to_string(k) + ")" + ext);
  return p;
}

// ai: moved, or copied then the store's removed where a rename cannot cross file systems; "" or why, the store's copy
// ai: then left as it was and no part of a copy left at to (to was a free name)
std::string moveOut(const fs::path& from, const fs::path& to) {
  std::error_code e;
  fs::rename(from, to, e);
  if (!e) return "";
  std::error_code c;
  fs::copy_file(from, to, c);
  if (c) {
    std::error_code r;
    fs::remove(to, r);
    return "the file could not be moved to " + to.u8string() + ": " + c.message();
  }
  fs::remove(from, c);
  return "";
}

std::string percent(double share) {
  char b[16];
  std::snprintf(b, sizeof b, "%.0f%%", share * 100);
  return b;
}

void lineOf(const json& j) {
  const json& c = j.contains("camera") ? j["camera"] : json();
  std::string tail;
  if (j.contains("file") && j["file"].is_object()) tail = ", file " + percent(num(j["file"], "fraction")) + " " + text(j["file"], "name");
  std::printf("rx: %s, camera %.1f fps, processed %.0f fps, found %s, %.0f blocks, %.0f KB/s%s\n", text(j, "state").c_str(), num(c, "fps"),
              num(j, "processedFps"), percent(num(j, "foundShare")).c_str(), num(j, "blocks"), num(j, "goodputKBs"), tail.c_str());
}

// ai: The run's share of frames found (the summary's: the stats' foundShare is the last second's, none in a run
// ai: shorter than its first window): the series (receiver.h, the newest 150 frames) read every poll from 2 s before
// ai: the newest frame counted, as its rows can come out of capture order (the CPU pool); a frame is its capture's ms,
// ai: the two halves of a 2:1 capture two rows of one ms, each ms's fullest view kept
struct Found {
  std::map<double, std::pair<int, int>> rows;   // ai: ms: (frames, frames found)
  double newest = 0;
  void poll(lizard::RxSession& s) {
    const std::vector<double> v = s.series(newest - 2000);
    std::map<double, std::pair<int, int>> snap;
    for (size_t i = 1; i + 8 <= v.size(); i += 8) {
      if (v[i] <= 0) continue;   // ai: a frame whose stamp the receiver no longer held: no ms to count it by
      auto& x = snap[v[i]];
      x.first++;
      x.second += v[i + 3] > 0;
    }
    for (const auto& [ms, x] : snap) {
      auto& y = rows[ms];
      if (x.first > y.first) y = x;
      newest = std::max(newest, ms);
    }
  }
  std::string said() const {
    int n = 0, found = 0;
    for (const auto& r : rows) {
      n += r.second.first;
      found += r.second.second;
    }
    return (n ? percent(double(found) / n) : std::string("none")) + " of " + std::to_string(n) + " frames";
  }
};

// ai: the run's last word (main's header): t since the camera started; the file's part where one came whole
void summary(const json& j, const Found& found, double t, bool whole) {
  const json& f = j.contains("file") && j["file"].is_object() ? j["file"] : json();
  const json& tot = j.contains("totals") ? j["totals"] : json();
  std::string file;
  if (whole) {
    char b[160];
    const double secs = num(f, "secs");
    std::snprintf(b, sizeof b, ", the file %.0f B in %.2f s (%.0f KB/s)", num(f, "size"), secs, secs > 0 ? num(f, "size") / secs / 1000 : 0.0);
    file = b;
  }
  std::printf("summary: %.2f s%s, found %s, %.0f blocks verified, %.0f frames decoded\n", t, file.c_str(), found.said().c_str(), num(tot, "blocks"),
              num(tot, "frames"));
}

int list() {
  std::vector<std::string> notes;
  std::vector<lizard::cam::Info> cams;
  try {
    cams = lizard::cam::list(&notes);
  } catch (const std::exception& e) {
    notes.push_back(std::string("the cameras could not be listed: ") + e.what());
  }
  for (const auto& i : cams) {
    std::string why;
    std::string line;
    try {
      auto c = lizard::cam::open(i.id, why);
      if (c) line = lizard::describeCamera(*c);
    } catch (const std::exception& e) {
      why = e.what();
    }
    if (line.empty())
      line = dump({{"info", {{"id", i.id}, {"name", i.name}, {"backend", i.backend}, {"detail", i.detail}}},
                   {"error", why.empty() ? "the camera could not be opened" : why}});
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
  }
  std::printf("%s\n", dump({{"notes", notes}}).c_str());
  return 0;
}

// ai: a fresh folder of our own in the system's temp folder
fs::path tempStore() {
  std::random_device rd;
  std::error_code e;
  const fs::path base = fs::temp_directory_path(e);
  for (int k = 0; k < 100; k++) {
    char b[32];
    std::snprintf(b, sizeof b, "lizard-receive-%08x", unsigned(rd()));
    const fs::path p = base / b;
    if (fs::create_directory(p, e)) return p;
  }
  throw std::runtime_error("no folder could be made in " + base.u8string() + (e ? ": " + e.message() : std::string()));
}

}  // namespace

static int run(int argc, char** argv) {
  const Args a = parse(argc, argv);
  if (a.list) return list();
  std::signal(SIGINT, [](int) { gQuit = true; });
  std::signal(SIGTERM, [](int) { gQuit = true; });
#if defined(SIGPIPE)
  // ai: stdout closed under it (a reader that stopped reading) ends no transfer, and the temp store is still removed
  std::signal(SIGPIPE, SIG_IGN);
#endif
  fs::path ours;   // ai: the temp store we made, removed at the end unless it holds the only copy of a file (keep)
  bool keep = false;
  int status = 1;
  {
    std::unique_ptr<lizard::RxSession> s;
    try {
      std::string why;
      const auto fail = [&](const std::string& w) { throw std::runtime_error(w.empty() ? std::string("refused") : w); };
      // ai: --out made (or found a folder) before anything streams, so a path that cannot be one fails at once, not
      // ai: after the whole file came
      {
        std::error_code e;
        const fs::path out = fs::u8path(a.out);
        fs::create_directories(out, e);
        if (!fs::is_directory(out)) fail("--out " + a.out + " is not a folder that can be made" + (e ? ": " + e.message() : std::string()));
      }
      fs::path store = fs::u8path(a.store);
      if (a.store.empty()) store = ours = tempStore();
      lizard::RxSession::Config c;
      c.assets = a.assets;
      c.storeDir = store.u8string();
      c.decoder = a.decoder;
      c.precision = a.precision;
      c.layout = a.layout;
      c.log = [](const std::string& m) { std::fprintf(stderr, "lizard: %s\n", m.c_str()); };
      s = lizard::RxSession::create(c, why);
      if (!s) fail(why);
      std::string id = a.camera;
      // ai: the simulated camera where one is set up (LIZ_CAMERA_SIM), with nothing listed: a test's run never reaches a
      // ai: real camera by default (2026-10-25, as V4L2 came to probe /dev/video* unless LIZ_V4L2_NODES says otherwise)
      if (id.empty() && std::getenv("LIZ_CAMERA_SIM")) id = "sim:0";
      if (id.empty()) {
        const json l = parsed(lizard::RxSession::cameras());
        if (!l.contains("cameras") || !l["cameras"].is_array() || l["cameras"].empty()) {
          std::string notes;
          if (l.contains("notes") && l["notes"].is_array())
            for (const auto& n : l["notes"]) notes += (notes.empty() ? " (" : "; ") + n.get<std::string>();
          fail("no camera was found" + (notes.empty() ? std::string() : notes + ")"));
        }
        id = text(l["cameras"][0], "id");
      }
      if (!s->open(id, why)) fail(why);
      const json cam = parsed(s->camera());
      const int mode = modeOf(cam.value("modes", json::array()), a.mode, int(num(cam, "pick")), why);
      if (mode < 0) fail(why);
      if (!s->start(mode, why)) fail(why);
      for (const auto& kv : a.sets) {
        const std::string k = kv.substr(0, kv.find('=')), v = kv.substr(kv.find('=') + 1);
        double value = 0;
        const bool automatic = v == "auto";
        if (!automatic) {
          try {
            value = std::stod(v);
          } catch (const std::logic_error&) { fail("--set " + kv + ": not a number"); }
        }
        if (!s->set(k, value, automatic, why)) fail("--set " + kv + ": " + why);
      }
      if (a.triggerMs > 0 && !s->trigger(std::llround(a.triggerMs * 1e6), why)) fail(why);
      const json m = cam["modes"][size_t(mode)];
      const json st = parsed(s->stats());
      char trig[64] = "";
      if (a.triggerMs > 0) std::snprintf(trig, sizeof trig, ", a software trigger every %.3f ms", a.triggerMs);
      std::printf("camera: %s (%s), mode %d: %.0f x %.0f %s at %.2f fps%s; decoder %s, %s; store %s\n", text(cam["info"], "name").c_str(), id.c_str(), mode,
                  num(m, "w"), num(m, "h"), text(m, "format").c_str(), num(m, "fps"), trig, text(st, "decoder").c_str(), a.layout.c_str(),
                  store.u8string().c_str());
      std::fflush(stdout);

      const auto t0 = std::chrono::steady_clock::now();
      double next = 1;
      Found found;
      for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        found.poll(*s);
        const std::string f = s->file();
        if (!f.empty()) {
          const fs::path from = fs::u8path(f);
          std::error_code e;
          const fs::path to = freeName(fs::u8path(a.out), from.filename());
          const std::string w = moveOut(from, to);
          // ai: the store holds the only copy of the file then: kept, and where it is said (2026-10-25: the temp store
          // ai: was removed with it)
          if (!w.empty()) {
            keep = true;
            fail(w + "; the file is kept at " + from.u8string());
          }
          const json j = parsed(s->stats());
          if (a.json) std::printf("%s\n", dump(j).c_str());
          else lineOf(j);
          found.poll(*s);
          summary(j, found, secsSince(t0), true);
          std::printf("file %s\n", fs::absolute(to, e).u8string().c_str());
          status = 0;
          break;
        }
        const double t = secsSince(t0);
        if (t >= next) {
          const json j = parsed(s->stats());
          if (a.json) std::printf("%s\n", dump(j).c_str());
          else lineOf(j);
          std::fflush(stdout);
          next += 1;
        }
        if (!s->streaming()) {
          const json j = parsed(s->stats());
          const std::string ended = j.contains("camera") ? text(j["camera"], "ended") : "";
          fail("the camera stopped: " + (ended.empty() ? std::string("the stream ended") : ended));
        }
        if (gQuit || (a.secs > 0 && t >= a.secs)) {
          const json j = parsed(s->stats());
          found.poll(*s);
          summary(j, found, t, false);
          status = j.contains("totals") && num(j["totals"], "blocks") > 0 ? 0 : 1;
          break;
        }
      }
    } catch (const std::exception& e) {
      std::printf("FAIL %s\n", e.what());
      status = 1;
    }
    std::fflush(stdout);
    s.reset();
  }
  if (!ours.empty() && !keep) {
    std::error_code e;
    fs::remove_all(ours, e);
  }
  return status;
}

#if defined(_WIN32)
// ai: Windows (2026-10-25): the arguments in UTF-16 (wmain, MinGW's -municode in CMakeLists.txt), each made UTF-8 for
// ai: run; main's came in the ANSI code page, so a --out, --store or --assets path with a character outside it did not
// ai: round-trip. windows.h here alone, after everything else, so its macros reach nothing above.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

int wmain(int argc, wchar_t** wargv) {
  std::vector<std::string> args(size_t(argc > 0 ? argc : 0));
  std::vector<char*> argv(args.size() + 1, nullptr);
  for (size_t i = 0; i < args.size(); i++) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
    if (n > 1) {
      args[i].resize(size_t(n));
      WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, args[i].data(), n, nullptr, nullptr);
      args[i].resize(size_t(n) - 1);   // ai: less the terminating zero
    }
    argv[i] = args[i].data();
  }
  return run(argc, argv.data());
}
#else
int main(int argc, char** argv) { return run(argc, argv); }
#endif
