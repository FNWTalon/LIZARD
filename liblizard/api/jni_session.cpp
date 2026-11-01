// ai: The receive session's JNI (2026-10-31): dev.lizard.SessionNative's static methods over lizard.h layer 5
// ai: (liz_session_*), registered by registerSession from the loading library's JNI_OnLoad (the binding's liblizard,
// ai: the desktop app's library). No callback into the JVM: every native thread (the session's, the camera's, the
// ai: receiver's) stays C++ alone, and the app polls (stats, camera, preview); the session's log goes to stderr (Android:
// ai: the system log, tag lizard). Strings are true UTF-8 and UTF-16 (jni_text.h). create's refusal is an
// ai: IllegalArgumentException with its sentence; the other calls return at once (lizard.h), the static ones that can
// ai: refuse with their sentence ("" none).
#include "jni_session.h"

#include <cstdio>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#include <android/native_window_jni.h>
#endif

#include "jni_text.h"
#include "lizard.h"

namespace lizard::jni {

namespace {

liz_session* ss(jlong h) { return reinterpret_cast<liz_session*>(h); }

void logLine(void*, const char* line) {
#if defined(__ANDROID__)
  __android_log_write(ANDROID_LOG_INFO, "lizard", line);
#else
  std::fprintf(stderr, "lizard: %s\n", line);
#endif
}

// ai: a JSON the C API writes into a buffer of the caller's: asked again with the length it said where that did not fit
// ai: (a stats may grow between two asks: a few tries)
template <class F>
std::string grown(F f) {
  std::string b(4096, '\0');
  for (int k = 0;; k++) {
    const size_t n = f(&b[0], b.size());
    if (n < b.size() || k == 3) {
      b.resize(n < b.size() ? n : b.size() - 1);
      return b;
    }
    b.assign(n + 1024, '\0');
  }
}

jboolean loaded(JNIEnv*, jclass) { return JNI_TRUE; }

jlong create(JNIEnv* e, jclass, jstring assets, jstring cacheDir, jstring dataDir, jstring decoder, jstring layout, jboolean forgetSettings) {
  const std::string a = str(e, assets), c = str(e, cacheDir), d = str(e, dataDir), dec = str(e, decoder), lay = str(e, layout);
  liz_session_config cfg{};
  cfg.assets = a.empty() ? nullptr : a.c_str();
  cfg.cache_dir = c.empty() ? nullptr : c.c_str();
  cfg.data_dir = d.c_str();
  cfg.decoder = dec.c_str();
  cfg.layout = lay.c_str();
  cfg.forget_settings = forgetSettings == JNI_TRUE;
  cfg.log = logLine;
  liz_session* s = liz_session_new(&cfg);
  if (!s) {
    e->ThrowNew(e->FindClass("java/lang/IllegalArgumentException"), liz_last_error());
    return 0;
  }
  return reinterpret_cast<jlong>(s);
}
void freeSession(JNIEnv*, jclass, jlong h) { liz_session_free(ss(h)); }

jstring cameras(JNIEnv* e, jclass) { return jstr(e, grown([](char* b, size_t n) { return liz_cameras(b, n); })); }
jstring camerasStamp(JNIEnv* e, jclass) { return jstr(e, grown([](char* b, size_t n) { return liz_cameras_stamp(b, n); })); }

void open(JNIEnv* e, jclass, jlong h, jstring id) { liz_session_open(ss(h), str(e, id).c_str()); }
void closeCamera(JNIEnv*, jclass, jlong h) { liz_session_close(ss(h)); }
void set(JNIEnv* e, jclass, jlong h, jstring control, jdouble value, jboolean automatic) {
  liz_session_set(ss(h), str(e, control).c_str(), value, automatic == JNI_TRUE);
}
void start(JNIEnv* e, jclass, jlong h, jstring mode) { liz_session_start(ss(h), str(e, mode).c_str()); }
void stop(JNIEnv*, jclass, jlong h) { liz_session_stop(ss(h)); }
void reconfigure(JNIEnv* e, jclass, jlong h, jstring decoder, jstring layout) {
  liz_session_reconfigure(ss(h), str(e, decoder).c_str(), str(e, layout).c_str());
}
void batchCap(JNIEnv*, jclass, jlong h, jint n) { liz_session_batch_cap(ss(h), n); }

jstring stats(JNIEnv* e, jclass, jlong h) { return jstr(e, grown([h](char* b, size_t n) { return liz_session_stats(ss(h), b, n); })); }
jstring camera(JNIEnv* e, jclass, jlong h) { return jstr(e, grown([h](char* b, size_t n) { return liz_session_camera(ss(h), b, n); })); }

// ai: The newest crop newer than `since` (a negative since: any), at most `side` px on its long side and never more than
// ai: `out` holds, its rows of w into out: [seq, w, h], or null.
jlongArray preview(JNIEnv* e, jclass, jlong h, jlong since, jint side, jbyteArray out) {
  if (!out || side <= 0) return nullptr;
  const jsize room = e->GetArrayLength(out);
  std::vector<uint8_t> px(size_t(room > 0 ? room : 0));
  uint32_t w = 0, hh = 0;
  uint64_t seq = 0;
  if (px.empty() || liz_session_preview(ss(h), since < 0 ? 0 : uint64_t(since), side, px.data(), px.size(), &w, &hh, &seq) != 1) return nullptr;
  if (int64_t(w) * hh > room) return nullptr;
  e->SetByteArrayRegion(out, 0, jsize(size_t(w) * hh), reinterpret_cast<const jbyte*>(px.data()));
  const jlong v[3] = {jlong(seq), jlong(w), jlong(hh)};
  jlongArray a = e->NewLongArray(3);
  if (a) e->SetLongArrayRegion(a, 0, 3, v);
  return a;
}

void forget(JNIEnv* e, jclass, jlong h, jstring root) { liz_session_forget(ss(h), str(e, root).c_str()); }
void clear(JNIEnv*, jclass, jlong h) { liz_session_clear(ss(h)); }
void replays(JNIEnv* e, jclass, jlong h, jboolean on, jstring meta) { liz_session_replays(ss(h), on == JNI_TRUE, str(e, meta).c_str()); }
void extra(JNIEnv* e, jclass, jlong h, jstring json) { liz_session_extra(ss(h), str(e, json).c_str()); }

// ai: The Android hooks (lizard.h): "" done, else why. previewWindow takes an android.view.Surface (null drops it)
// ai: and waits for the camera to take it or let it go: never on the UI thread.
jstring why(JNIEnv* e, int rc) { return jstr(e, rc == LIZ_OK ? std::string() : std::string(liz_last_error())); }
jstring previewWindow(JNIEnv* e, jclass, jlong h, jobject surface) {
#if defined(__ANDROID__)
  ANativeWindow* w = surface ? ANativeWindow_fromSurface(e, surface) : nullptr;
  if (surface && !w) return jstr(e, "the surface has no window");
  const int rc = liz_session_preview_window(ss(h), w);
  if (w) ANativeWindow_release(w);
  return why(e, rc);
#else
  (void)surface;
  return why(e, liz_session_preview_window(ss(h), nullptr));
#endif
}
// ai: the PC link's, process-wide (no session)
jstring link(JNIEnv* e, jclass, jboolean up, jstring layout) { return why(e, liz_link(up == JNI_TRUE, str(e, layout).c_str())); }
jstring linkPack(JNIEnv* e, jclass, jboolean on) { return why(e, liz_link_pack(on == JNI_TRUE)); }
jstring linkState(JNIEnv* e, jclass) { return jstr(e, grown([](char* b, size_t n) { return liz_link_state(b, n); })); }
jstring usbAccessory(JNIEnv* e, jclass, jint fd) { return why(e, liz_usb_accessory(fd)); }

jstring kept(JNIEnv* e, jclass, jstring dataDir) {
  const std::string d = str(e, dataDir);
  return jstr(e, grown([&d](char* b, size_t n) { return liz_kept(d.c_str(), b, n); }));
}
jstring placed(JNIEnv* e, jclass, jstring dataDir, jstring id) {
  return jstr(e, liz_kept_placed(str(e, dataDir).c_str(), str(e, id).c_str()) == LIZ_OK ? std::string() : std::string(liz_last_error()));
}
jstring partials(JNIEnv* e, jclass, jstring dataDir) {
  const std::string d = str(e, dataDir);
  return jstr(e, grown([&d](char* b, size_t n) { return liz_partials(d.c_str(), b, n); }));
}
jstring deletePartial(JNIEnv* e, jclass, jstring dataDir, jstring id) {
  return jstr(e, liz_partial_delete(str(e, dataDir).c_str(), str(e, id).c_str()) == LIZ_OK ? std::string() : std::string(liz_last_error()));
}

#define M(name, sig) {const_cast<char*>(#name), const_cast<char*>(sig), reinterpret_cast<void*>(name)}
#define S "Ljava/lang/String;"
const JNINativeMethod METHODS[] = {
    M(loaded, "()Z"),
    M(create, "(" S S S S S "Z)J"),
    {const_cast<char*>("free"), const_cast<char*>("(J)V"), reinterpret_cast<void*>(freeSession)},
    M(cameras, "()" S),
    M(camerasStamp, "()" S),
    M(open, "(J" S ")V"),
    M(closeCamera, "(J)V"),
    M(set, "(J" S "DZ)V"),
    M(start, "(J" S ")V"),
    M(stop, "(J)V"),
    M(reconfigure, "(J" S S ")V"),
    M(batchCap, "(JI)V"),
    M(stats, "(J)" S),
    M(camera, "(J)" S),
    M(preview, "(JJI[B)[J"),
    M(forget, "(J" S ")V"),
    M(clear, "(J)V"),
    M(replays, "(JZ" S ")V"),
    M(extra, "(J" S ")V"),
    M(previewWindow, "(JLjava/lang/Object;)" S),
    M(link, "(Z" S ")" S),
    M(linkPack, "(Z)" S),
    M(linkState, "()" S),
    M(usbAccessory, "(I)" S),
    M(kept, "(" S ")" S),
    M(placed, "(" S S ")" S),
    M(partials, "(" S ")" S),
    M(deletePartial, "(" S S ")" S),
};
#undef S
#undef M

}  // namespace

bool registerSession(JNIEnv* e) {
  jclass c = e->FindClass("dev/lizard/SessionNative");
  if (!c) return false;
  return e->RegisterNatives(c, METHODS, sizeof METHODS / sizeof *METHODS) == JNI_OK;
}

}  // namespace lizard::jni
