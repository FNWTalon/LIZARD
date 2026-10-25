// ai: The receiver's JNI face (2026-10-25): dev.lizard.desktop.Native's rx functions over an RxSession (rx_session.h),
// ai: whose pointer is the handle. No callback into the JVM: every native thread (the camera's capture thread, the
// ai: receiver's, the trigger's schedule) stays C++ alone, and the app polls (rxStats, rxPreview) or blocks a thread of
// ai: its own (rxCaptures, the phase lock's). No C++ exception crosses into the JVM: a refusal comes back as its
// ai: sentence ("" none), rxCreate's and rxReplayNew's through txError's channel (jni_util.h) with 0. Strings are
// ai: jni_util.h's true UTF-8 and UTF-16. The receiver's log goes to stderr, as the sender's presenter's does.
#include <jni.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "jni_util.h"
#include "json.hpp"
#include "replay.h"
#include "rx_session.h"

using lizard::jni::jstr;
using lizard::jni::setError;
using lizard::jni::str;

namespace {

lizard::RxSession* rs(jlong h) { return reinterpret_cast<lizard::RxSession*>(h); }
// ai: Save replays: a handle is a shared_ptr to the replay (Android's jni.cpp), the receiver holding another while it
// ai: takes frames, so the run outlives either; rxReplayEnd lets the handle's go
std::shared_ptr<lizard::Replay>* replayOf(jlong r) { return reinterpret_cast<std::shared_ptr<lizard::Replay>*>(r); }

void logLine(const std::string& m) { std::fprintf(stderr, "lizard: %s\n", m.c_str()); }

// ai: f's refusal ("" none), an exception as its sentence
template <class F>
std::string refusal(F f) {
  try {
    return f();
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "the receiver failed";
  }
}

// ai: a call with nothing to answer: an exception goes to the log
template <class F>
void quietly(const char* what, F f) {
  try {
    f();
  } catch (const std::exception& e) {
    logLine(std::string(what) + ": " + e.what());
  } catch (...) {
    logLine(std::string(what) + " failed");
  }
}

jdoubleArray doubles(JNIEnv* e, const std::vector<double>& v) {
  jdoubleArray a = e->NewDoubleArray(jsize(v.size()));
  if (a && !v.empty()) e->SetDoubleArrayRegion(a, 0, jsize(v.size()), v.data());
  return a;   // ai: null only with the VM's OutOfMemoryError pending
}

jlongArray longs(JNIEnv* e, const std::vector<jlong>& v) {
  jlongArray a = e->NewLongArray(jsize(v.size()));
  if (a && !v.empty()) e->SetLongArrayRegion(a, 0, jsize(v.size()), v.data());
  return a;
}

}  // namespace

FN(jstring, rxCameras)(JNIEnv* e, jclass) {
  std::string s;
  try {
    s = lizard::RxSession::cameras();
  } catch (const std::exception& x) {
    s = nlohmann::json{{"cameras", nlohmann::json::array()},
                       {"notes", nlohmann::json::array({std::string("the cameras could not be listed: ") + x.what()})}}
            .dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
  }
  return jstr(e, s);
}

// ai: decoder auto|gpu|cpu, precision auto|int8|f16|f32, layout 1:1|2:1; storeDir the transfer's (its lizard-xfer
// ai: folder kept across sessions to resume); cacheDir the pipeline cache's ("" none). 0 with why in txError().
FN(jlong, rxCreate)(JNIEnv* e, jclass, jstring assets, jstring cacheDir, jstring storeDir, jstring decoder, jstring precision, jstring layout) {
  lizard::RxSession::Config c;
  c.assets = str(e, assets);
  c.cacheDir = str(e, cacheDir);
  c.storeDir = str(e, storeDir);
  c.decoder = str(e, decoder);
  c.precision = str(e, precision);
  c.layout = str(e, layout);
  c.log = logLine;
  try {
    std::string why;
    auto s = lizard::RxSession::create(c, why);
    if (!s) {
      setError(why.empty() ? "the receiver could not be made" : why);
      return 0;
    }
    return reinterpret_cast<jlong>(s.release());
  } catch (const std::exception& x) {
    setError(std::string("the receiver could not be made: ") + x.what());
  } catch (...) {
    setError("the receiver could not be made");
  }
  return 0;
}

FN(jstring, rxConfigure)(JNIEnv* e, jclass, jlong h, jstring decoder, jstring layout) {
  if (!h) return jstr(e, "no receiver");
  const std::string d = str(e, decoder), l = str(e, layout);
  return jstr(e, refusal([&] {
                std::string why;
                return rs(h)->reconfigure(d, l, why) ? std::string() : why.empty() ? std::string("the receiver could not be changed") : why;
              }));
}

FN(jstring, rxOpen)(JNIEnv* e, jclass, jlong h, jstring id) {
  if (!h) return jstr(e, "no receiver");
  const std::string i = str(e, id);
  return jstr(e, refusal([&] {
                std::string why;
                return rs(h)->open(i, why) ? std::string() : why.empty() ? "the camera " + i + " could not be opened" : why;
              }));
}

FN(void, rxClose)(JNIEnv*, jclass, jlong h) {
  if (h) quietly("camera close", [&] { rs(h)->close(); });
}

FN(jstring, rxCamera)(JNIEnv* e, jclass, jlong h) {
  std::string s = "{}";
  if (h) quietly("camera", [&] { s = rs(h)->camera(); });
  return jstr(e, s);
}

FN(jstring, rxSet)(JNIEnv* e, jclass, jlong h, jstring control, jdouble value, jboolean automatic) {
  if (!h) return jstr(e, "no receiver");
  const std::string c = str(e, control);
  return jstr(e, refusal([&] {
                std::string why;
                return rs(h)->set(c, value, automatic == JNI_TRUE, why) ? std::string() : why.empty() ? "the camera refused " + c : why;
              }));
}

FN(jstring, rxStart)(JNIEnv* e, jclass, jlong h, jint mode) {
  if (!h) return jstr(e, "no receiver");
  return jstr(e, refusal([&] {
                std::string why;
                return rs(h)->start(mode, why) ? std::string() : why.empty() ? std::string("the camera did not start") : why;
              }));
}

FN(void, rxStop)(JNIEnv*, jclass, jlong h) {
  if (h) quietly("camera stop", [&] { rs(h)->stop(); });
}

FN(jstring, rxStats)(JNIEnv* e, jclass, jlong h) {
  if (!h) return jstr(e, "{}");
  std::string s;
  try {
    s = rs(h)->stats();
  } catch (const std::exception& x) {
    s = nlohmann::json{{"state", "error"}, {"error", std::string("the stats could not be read: ") + x.what()}, {"camera", nullptr}}.dump(
        -1, ' ', false, nlohmann::json::error_handler_t::replace);
  }
  return jstr(e, s);
}

// ai: Receiver::series as a double[], asked at every capture by the phase lock: no JSON on that path
FN(jdoubleArray, rxSeries)(JNIEnv* e, jclass, jlong h, jdouble sinceMs) {
  std::vector<double> v{0};
  if (h) quietly("series", [&] { v = rs(h)->series(sinceMs); });
  return doubles(e, v);
}

// ai: blocks this thread (the lock's) up to timeoutMs: (ns, delayed) pairs
FN(jlongArray, rxCaptures)(JNIEnv* e, jclass, jlong h, jint timeoutMs) {
  std::vector<jlong> v;
  if (h) quietly("captures", [&] {
    for (const int64_t x : rs(h)->captures(timeoutMs)) v.push_back(jlong(x));
  });
  return longs(e, v);
}

FN(void, rxWake)(JNIEnv*, jclass, jlong h) {
  if (h) quietly("wake", [&] { rs(h)->wake(); });
}

FN(jstring, rxTrigger)(JNIEnv* e, jclass, jlong h, jlong periodNs) {
  if (!h) return jstr(e, "no receiver");
  return jstr(e, refusal([&] {
                std::string why;
                return rs(h)->trigger(int64_t(periodNs), why) ? std::string() : why.empty() ? std::string("the camera refused its trigger") : why;
              }));
}

FN(void, rxDelay)(JNIEnv*, jclass, jlong h, jlong ns) {
  if (h) quietly("delay", [&] { rs(h)->delay(int64_t(ns)); });
}

// ai: The newest crop newer than `since` (a negative since: any), at most `side` px on its long side and never more
// ai: than `out` holds (the side shrunk to fit a short array), its rows of w into out: [seq, w, h], or null.
FN(jlongArray, rxPreview)(JNIEnv* e, jclass, jlong h, jlong since, jint side, jbyteArray out) {
  if (!h || !out || side <= 0) return nullptr;
  const jsize room = e->GetArrayLength(out);
  int s = side;
  if (jlong(s) * s > room) s = int(std::floor(std::sqrt(double(room))));
  if (s <= 0) return nullptr;
  std::vector<uint8_t> px;
  uint32_t w = 0, ht = 0;
  uint64_t seq = 0;
  bool got = false;
  quietly("preview", [&] { got = rs(h)->preview(since < 0 ? 0 : uint64_t(since), s, px, w, ht, seq); });
  if (!got || jlong(w) * ht > room || px.size() < size_t(w) * ht) return nullptr;
  e->SetByteArrayRegion(out, 0, jsize(size_t(w) * ht), reinterpret_cast<const jbyte*>(px.data()));
  return longs(e, {jlong(seq), jlong(w), jlong(ht)});
}

FN(jstring, rxFile)(JNIEnv* e, jclass, jlong h) {
  std::string s;
  if (h) quietly("file", [&] { s = rs(h)->file(); });
  return jstr(e, s);
}

// ai: waits for the transfer's worker (receiver.h clear): not on the UI thread
FN(void, rxClear)(JNIEnv*, jclass, jlong h) {
  if (h) quietly("clear", [&] { rs(h)->clear(); });
}

// ai: one transfer by its root or its folder's id (receiver.h forget); waits for the transfer's worker
FN(void, rxForget)(JNIEnv* e, jclass, jlong h, jstring root) {
  if (!h || !root) return;
  const std::string r = str(e, root);
  quietly("forget", [&] { rs(h)->forget(r); });
}

FN(void, rxBatchCap)(JNIEnv*, jclass, jlong h, jint n) {
  if (h) quietly("batch cap", [&] { rs(h)->batchCap(n); });
}

// ai: a run's folder (made here: one that holds anything is refused, which rxReplayEnd's error says), the newest
// ai: `frames` frames kept; 0 with why in txError() where it could not be made at all
FN(jlong, rxReplayNew)(JNIEnv* e, jclass, jstring dir, jstring run, jint frames) {
  try {
    return reinterpret_cast<jlong>(new std::shared_ptr<lizard::Replay>(std::make_shared<lizard::Replay>(str(e, dir), str(e, run), frames)));
  } catch (const std::exception& x) {
    setError(std::string("the replay could not be made: ") + x.what());
  } catch (...) {
    setError("the replay could not be made");
  }
  return 0;
}

// ai: replay 0: no more frames to a replay
FN(void, rxRecord)(JNIEnv*, jclass, jlong h, jlong replay) {
  if (h) quietly("record", [&] { rs(h)->record(replay ? *replayOf(replay) : nullptr); });
}

// ai: The run's end (Replay::finish, which waits up to a few seconds for frames the GPU decoder still holds for it),
// ai: the handle let go: {"frames", "bytes", "error"}, frames -1 where the folder, a rename or the meta failed (error
// ai: says why), error otherwise the write that ended the frames early ("" none). Not on the UI thread.
FN(jstring, rxReplayEnd)(JNIEnv* e, jclass, jlong replay, jstring rxStats, jstring rows, jstring more) {
  nlohmann::json o = {{"frames", -1}, {"bytes", 0}, {"error", "no replay"}};
  if (replay) {
    std::shared_ptr<lizard::Replay>* p = replayOf(replay);
    const std::string s = str(e, rxStats), r = str(e, rows), m = str(e, more);
    try {
      const int n = (*p)->finish(s, r, m);
      o = {{"frames", n}, {"bytes", (*p)->bytes()}, {"error", (*p)->error()}};
    } catch (const std::exception& x) {
      o = {{"frames", -1}, {"bytes", 0}, {"error", std::string("the replay could not be finished: ") + x.what()}};
    }
    delete p;
  }
  return jstr(e, o.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
}

// ai: The camera stopped and closed, the lock's waits ended, then the receiver (rx_session.h): no other rx call on
// ai: this handle during or after it (the lock's thread woken and joined first).
FN(void, rxDestroy)(JNIEnv*, jclass, jlong h) {
  if (h) quietly("destroy", [&] { delete rs(h); });
}
