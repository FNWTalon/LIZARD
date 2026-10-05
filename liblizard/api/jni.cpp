// ai: The Kotlin binding's native half (2026-10-03; bindings/kotlin): the C API (lizard.h) for dev.lizard.Native's
// ai: static methods, registered in JNI_OnLoad (so no Java_ names, and the library loads in a JVM or on Android alike).
// ai: A failed call throws: LIZ_E_ARG IllegalArgumentException, LIZ_E_STATE IllegalStateException, LIZ_E_UNSUPPORTED
// ai: UnsupportedOperationException, LIZ_E_NOMEM OutOfMemoryError, anything else dev.lizard.LizardException(code,
// ai: message). Every array and direct buffer is checked against what the call reads or writes first (holds, holdsBuf);
// ai: arrays are read in place (GetPrimitiveArrayCritical) only for a short call that makes no JNI call meanwhile, and
// ai: pinned or copied (Elems, GetByteArrayRegion) for one that calls back into Java or runs long.
#include <jni.h>

#include <cstring>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/hardware_buffer_jni.h>
#endif

#include "lizard.h"

namespace {

jclass gLizardException = nullptr;
jmethodID gLizardExceptionInit = nullptr;
JavaVM* gVm = nullptr;

// ai: A native thread's JNIEnv: attached once, detached when the thread ends (a thread_local's destructor; the engines'
// ai: threads call back into Kotlin: a frame released, a log line).
struct Attached {
  JNIEnv* env = nullptr;
  ~Attached() { if (env) gVm->DetachCurrentThread(); }
};
thread_local Attached tAttached;
JNIEnv* envHere() {
  JNIEnv* e = nullptr;
  if (gVm->GetEnv(reinterpret_cast<void**>(&e), JNI_VERSION_1_6) == JNI_OK) return e;
  if (tAttached.env) return tAttached.env;
#ifdef __ANDROID__
  if (gVm->AttachCurrentThread(&e, nullptr) != JNI_OK) return nullptr;
#else
  if (gVm->AttachCurrentThread(reinterpret_cast<void**>(&e), nullptr) != JNI_OK) return nullptr;
#endif
  tAttached.env = e;
  return e;
}

void throwFor(JNIEnv* e, int code) {
  const char* msg = liz_last_error();
  const char* cls = code == LIZ_E_ARG ? "java/lang/IllegalArgumentException"
                    : code == LIZ_E_STATE ? "java/lang/IllegalStateException"
                    : code == LIZ_E_UNSUPPORTED ? "java/lang/UnsupportedOperationException"
                    : code == LIZ_E_NOMEM ? "java/lang/OutOfMemoryError" : nullptr;
  if (cls) { e->ThrowNew(e->FindClass(cls), msg); return; }
  jstring m = e->NewStringUTF(msg);
  if (auto ex = static_cast<jthrowable>(e->NewObject(gLizardException, gLizardExceptionInit, jint(code), m))) e->Throw(ex);
}
// ai: a call's code: thrown where negative; the value back for the caller to return (ignored once thrown)
int ok(JNIEnv* e, int rc) { if (rc < 0) throwFor(e, rc); return rc; }
// ai: a handle made, or the call's failure thrown by its code
template <class T>
jlong made(JNIEnv* e, T* p) {
  if (!p) throwFor(e, liz_last_error_code() < 0 ? liz_last_error_code() : LIZ_E_INTERNAL);
  return reinterpret_cast<jlong>(p);
}

struct Str {
  JNIEnv* e; jstring s; const char* c;
  Str(JNIEnv* e, jstring s) : e(e), s(s), c(s ? e->GetStringUTFChars(s, nullptr) : nullptr) {}
  ~Str() { if (c) e->ReleaseStringUTFChars(s, c); }
};
// ai: an array's bytes in place for a call that makes no JNI call meanwhile
struct Crit {
  JNIEnv* e; jarray a; void* p;
  Crit(JNIEnv* e, jarray a) : e(e), a(a), p(a ? e->GetPrimitiveArrayCritical(a, nullptr) : nullptr) {}
  ~Crit() { if (p) e->ReleasePrimitiveArrayCritical(a, p, 0); }
  uint8_t* bytes() const { return static_cast<uint8_t*>(p); }
};
uint8_t* direct(JNIEnv* e, jobject buf) { return static_cast<uint8_t*>(e->GetDirectBufferAddress(buf)); }

// ai: A Kotlin array or direct buffer too short for what a call reads or writes, or an offset or stride that leaves it,
// ai: is an IllegalArgumentException before the C touches it (2026-10-03: a short `out` was
// ai: written past its end, which corrupted the JVM's heap). Counts in the array's elements, bytes for a buffer.
void throwArg(JNIEnv* e, const std::string& m) { e->ThrowNew(e->FindClass("java/lang/IllegalArgumentException"), m.c_str()); }
int bppOf(jint fmt) { return fmt == LIZ_GREY8 ? 1 : fmt == LIZ_RGBX8 || fmt == LIZ_RGBA8 || fmt == LIZ_BGRA8 ? 4 : 0; }
// ai: the bytes h rows of w pixels (bpp bytes each), rows stride apart, from offset, reach; -1 for a shape there cannot be
int64_t spanOf(int64_t offset, int64_t w, int64_t hh, int64_t stride, int bpp) {
  if (offset < 0 || w <= 0 || hh <= 0 || bpp <= 0 || stride < w * bpp) return -1;
  return offset + (hh - 1) * stride + w * bpp;
}
bool holds(JNIEnv* e, jarray a, int64_t need, const char* what) {
  const int64_t n = a ? e->GetArrayLength(a) : -1;
  if (need >= 0 && n >= need) return true;
  throwArg(e, std::string(what) + (need < 0 ? ": a shape there cannot be" : n < 0 ? ": null" : ": " + std::to_string(n) + " where " + std::to_string(need) + " are needed"));
  return false;
}
bool holdsBuf(JNIEnv* e, jobject b, int64_t need, const char* what) {
  const int64_t n = b ? e->GetDirectBufferCapacity(b) : -1;
  if (need >= 0 && n >= need && direct(e, b)) return true;
  throwArg(e, std::string(what) + (need < 0 ? ": a shape there cannot be" : n < 0 ? ": not a direct buffer" : ": " + std::to_string(n) + " bytes where " + std::to_string(need) + " are needed"));
  return false;
}
// ai: An array's bytes for a call that may call back into Java or run long (a receiver's push, a whole file's hashing):
// ai: pinned or copied by the VM, never a critical region, in which no JNI call may be made and the GC waits.
struct Elems {
  JNIEnv* e; jbyteArray a; jbyte* p;
  Elems(JNIEnv* e, jbyteArray a) : e(e), a(a), p(a ? e->GetByteArrayElements(a, nullptr) : nullptr) {}
  ~Elems() { if (p) e->ReleaseByteArrayElements(a, p, JNI_ABORT); }
  uint8_t* bytes() const { return reinterpret_cast<uint8_t*>(p); }
};

template <class T> T* h(jlong v) { return reinterpret_cast<T*>(v); }

jstring version(JNIEnv* e, jclass) { return e->NewStringUTF(liz_version()); }
jint abi(JNIEnv*, jclass) { return liz_abi(); }
jboolean simd(JNIEnv*, jclass) { return liz_simd() ? JNI_TRUE : JNI_FALSE; }
jint ringCells(JNIEnv* e, jclass, jint ring) { return ok(e, liz_ring_cells(ring)); }
void geometry(JNIEnv* e, jclass, jint blocks, jint ring, jint fps, jint codes, jintArray out) {
  liz_format f{blocks, ring, fps, codes};
  liz_geometry g{};
  if (ok(e, liz_geometry_of(&f, &g)) < 0) return;
  const jint v[8] = {g.n, g.span, g.pxm, g.side, g.width, g.height, g.gap, g.frame_blocks};
  e->SetIntArrayRegion(out, 0, 8, v);
}
jdouble roomFor(JNIEnv* e, jclass, jint blocks, jint ring) {
  const double v = liz_room_for(blocks, ring);
  if (v < 0) throwFor(e, int(v));
  return v;
}
jint pick(JNIEnv* e, jclass, jdouble w, jdouble hh, jint codes, jint ring, jint top) { return ok(e, liz_pick(w, hh, codes, ring, top)); }
void streamFill(JNIEnv* e, jclass, jint id, jbyteArray out) {
  uint8_t p[LIZ_PAYLOAD];
  liz_stream_fill(uint32_t(id), p);
  e->SetByteArrayRegion(out, 0, LIZ_PAYLOAD, reinterpret_cast<const jbyte*>(p));
}
jint layoutRects(JNIEnv* e, jclass, jint w, jint hh, jint layout, jintArray out) {
  int r[2][4];
  const int n = ok(e, liz_layout_rects(w, hh, layout, r));
  if (n > 0) e->SetIntArrayRegion(out, 0, 4 * n, &r[0][0]);
  return n;
}

jlong encoderNew(JNIEnv* e, jclass, jint blocks, jint ring, jint fps, jint codes) {
  liz_format f{blocks, ring, fps, codes};
  return made(e, liz_encoder_new(&f));
}
void encoderFree(JNIEnv*, jclass, jlong v) { liz_encoder_free(h<liz_encoder>(v)); }
void encoderPaint(JNIEnv* e, jclass, jlong v, jbyteArray blocks, jint picture, jbyteArray out, jint stride, jint fmt) {
  liz_geometry g{};
  if (ok(e, liz_encoder_geometry(h<liz_encoder>(v), &g)) < 0) return;
  if (!holds(e, blocks, int64_t(g.frame_blocks) * LIZ_BLOCK, "blocks") || !holds(e, out, spanOf(0, g.width, g.height, stride, bppOf(fmt)), "out")) return;
  int rc;
  {
    Crit b(e, blocks), o(e, out);
    rc = liz_encoder_paint(h<liz_encoder>(v), b.bytes(), uint32_t(picture), o.bytes(), stride, liz_pixfmt(fmt));
  }
  ok(e, rc);
}
void encoderPaintBuffer(JNIEnv* e, jclass, jlong v, jbyteArray blocks, jint picture, jobject out, jint stride, jint fmt) {
  liz_geometry g{};
  if (ok(e, liz_encoder_geometry(h<liz_encoder>(v), &g)) < 0) return;
  if (!holds(e, blocks, int64_t(g.frame_blocks) * LIZ_BLOCK, "blocks") || !holdsBuf(e, out, spanOf(0, g.width, g.height, stride, bppOf(fmt)), "out")) return;
  uint8_t* o = direct(e, out);
  int rc;
  { Crit b(e, blocks); rc = liz_encoder_paint(h<liz_encoder>(v), b.bytes(), uint32_t(picture), o, stride, liz_pixfmt(fmt)); }
  ok(e, rc);
}

jlong decoderNew(JNIEnv* e, jclass, jint nmax) { return made(e, liz_decoder_new(nmax)); }
void decoderFree(JNIEnv*, jclass, jlong v) { liz_decoder_free(h<liz_decoder>(v)); }
jint decoderMaxBlocks(JNIEnv* e, jclass, jlong v) { return ok(e, liz_decoder_max_blocks(h<liz_decoder>(v))); }
// ai: the decode's result into ints (found, ring, n, word, blocks, fps, held_used, total, verified, pilot_blocks) and
// ai: floats (quad[8], pilot_r[2], pilot_sd[2])
void putDecoded(JNIEnv* e, const liz_decoded& d, jintArray ints, jfloatArray floats) {
  const jint i[10] = {d.found, d.ring, d.n, d.word, d.blocks, d.fps, d.held_used, d.total, d.verified, d.pilot_blocks};
  float f[12];
  std::memcpy(f, d.quad, sizeof d.quad);
  std::memcpy(f + 8, d.pilot_r, sizeof d.pilot_r);
  std::memcpy(f + 10, d.pilot_sd, sizeof d.pilot_sd);
  e->SetIntArrayRegion(ints, 0, 10, i);
  e->SetFloatArrayRegion(floats, 0, 12, f);
}
// ai: the decode's outputs hold what it writes: verified the decoder's most blocks, held 1, ints 10, floats 12
bool decodeOuts(JNIEnv* e, jlong v, jintArray held, jbyteArray verified, jintArray ints, jfloatArray floats) {
  return holds(e, verified, int64_t(liz_decoder_max_blocks(h<liz_decoder>(v))) * LIZ_BLOCK, "verified") && holds(e, held, 1, "held") &&
         holds(e, ints, 10, "ints") && holds(e, floats, 12, "floats");
}
jint decodeCommon(JNIEnv* e, jlong v, const uint8_t* px, jint w, jint hh, jint stride, jint fmt, jintArray held, jbyteArray verified,
                  jintArray ints, jfloatArray floats) {
  jint hv = 0;
  e->GetIntArrayRegion(held, 0, 1, &hv);
  int heldWord = hv, rc;
  liz_decoded d{};
  { Crit out(e, verified); rc = liz_decode(h<liz_decoder>(v), px, w, hh, stride, liz_pixfmt(fmt), &heldWord, out.bytes(), &d); }
  if (ok(e, rc) < 0) return rc;
  hv = heldWord;
  e->SetIntArrayRegion(held, 0, 1, &hv);
  putDecoded(e, d, ints, floats);
  return rc;
}
jint decode(JNIEnv* e, jclass, jlong v, jbyteArray px, jint offset, jint w, jint hh, jint stride, jint fmt, jintArray held,
            jbyteArray verified, jintArray ints, jfloatArray floats) {
  if (!holds(e, px, spanOf(offset, w, hh, stride, bppOf(fmt)), "px") || !decodeOuts(e, v, held, verified, ints, floats)) return 0;
  jint heldWord = 0;
  e->GetIntArrayRegion(held, 0, 1, &heldWord);
  // ai: the image read in place: between here and the release no JNI call is made, the results written after it
  jbyte* p = static_cast<jbyte*>(e->GetPrimitiveArrayCritical(px, nullptr));
  int hw = heldWord, rc;
  liz_decoded d{};
  {
    Crit out(e, verified);
    rc = liz_decode(h<liz_decoder>(v), reinterpret_cast<uint8_t*>(p) + offset, w, hh, stride, liz_pixfmt(fmt), &hw, out.bytes(), &d);
  }
  e->ReleasePrimitiveArrayCritical(px, p, JNI_ABORT);
  if (ok(e, rc) < 0) return rc;
  heldWord = hw;
  e->SetIntArrayRegion(held, 0, 1, &heldWord);
  putDecoded(e, d, ints, floats);
  return rc;
}
jint decodeBuffer(JNIEnv* e, jclass, jlong v, jobject px, jint offset, jint w, jint hh, jint stride, jint fmt, jintArray held,
                  jbyteArray verified, jintArray ints, jfloatArray floats) {
  if (!holdsBuf(e, px, spanOf(offset, w, hh, stride, bppOf(fmt)), "px") || !decodeOuts(e, v, held, verified, ints, floats)) return 0;
  return decodeCommon(e, v, direct(e, px) + offset, w, hh, stride, fmt, held, verified, ints, floats);
}

jlong txNew(JNIEnv* e, jclass, jbyteArray bytes, jstring name, jstring type) {
  Str n(e, name), t(e, type);
  if (!holds(e, bytes, 0, "bytes")) return 0;
  const jsize len = e->GetArrayLength(bytes);
  liz_tx* tx;
  { Elems b(e, bytes); if (len && !b.p) return 0; tx = liz_tx_new(b.bytes(), size_t(len), n.c, t.c, LIZ_TX_COPY); }   // ai: hashes the whole file
  return made(e, tx);
}
jlong txNewPath(JNIEnv* e, jclass, jstring path, jstring name, jstring type) {
  Str p(e, path), n(e, name), t(e, type);
  return made(e, liz_tx_new_path(p.c, n.c, t.c));
}
jlong txNewTest(JNIEnv* e, jclass, jint first) { return made(e, liz_tx_new_test(uint32_t(first))); }
void txFree(JNIEnv*, jclass, jlong v) { liz_tx_free(h<liz_tx>(v)); }
jint txNext(JNIEnv* e, jclass, jlong v, jint n, jbyteArray out) {
  if (!holds(e, out, int64_t(n) * LIZ_BLOCK, "out")) return 0;
  int rc;
  { Crit o(e, out); rc = liz_tx_next(h<liz_tx>(v), n, o.bytes()); }
  return ok(e, rc);
}
void txInfo(JNIEnv* e, jclass, jlong v, jlongArray out, jbyteArray root) {
  liz_tx_info i{};
  if (ok(e, liz_tx_info_get(h<liz_tx>(v), &i)) < 0) return;
  const jlong l[5] = {i.test, jlong(i.length), jlong(i.chunks), jlong(i.lap), jlong(i.sent)};
  e->SetLongArrayRegion(out, 0, 5, l);
  e->SetByteArrayRegion(root, 0, 32, reinterpret_cast<const jbyte*>(i.root));
}

jlong rxNew(JNIEnv* e, jclass, jstring dir, jlong maxBytes) {
  Str d(e, dir);
  liz_store s{d.c, uint64_t(maxBytes)};
  return made(e, liz_rx_new(&s));
}
void rxFree(JNIEnv*, jclass, jlong v) { liz_rx_free(h<liz_rx>(v)); }
void rxFrame(JNIEnv* e, jclass, jlong v, jbyteArray blocks, jint count, jintArray out) {
  if (!holds(e, blocks, int64_t(count) * LIZ_BLOCK, "blocks") || !holds(e, out, 5, "out")) return;
  liz_verdict d{};
  // ai: copied out first: the frame may solve a chunk and write the file, too long a time to hold a critical region
  std::vector<uint8_t> b(size_t(count) * LIZ_BLOCK);
  if (count) e->GetByteArrayRegion(blocks, 0, jsize(b.size()), reinterpret_cast<jbyte*>(b.data()));
  const int rc = liz_rx_frame(h<liz_rx>(v), b.data(), count, &d);
  if (ok(e, rc) < 0) return;
  const jint i[5] = {d.seen, d.bad, d.judged, d.fresh, d.test};
  e->SetIntArrayRegion(out, 0, 5, i);
}
void rxProgress(JNIEnv* e, jclass, jlong v, jintArray ints, jlongArray longs, jdoubleArray doubles) {
  liz_progress p{};
  if (ok(e, liz_rx_progress(h<liz_rx>(v), &p)) < 0) return;
  const jint i[5] = {p.header, p.done, jint(p.chunks), jint(p.verified), jint(p.rejected)};
  const jlong l[4] = {jlong(p.length), jlong(p.bytes_in), jlong(p.sent), jlong(p.sent_in)};
  const jdouble d[2] = {p.fraction, p.solve_ms};
  e->SetIntArrayRegion(ints, 0, 5, i);
  e->SetLongArrayRegion(longs, 0, 4, l);
  e->SetDoubleArrayRegion(doubles, 0, 2, d);
}
jbyteArray rxChunks(JNIEnv* e, jclass, jlong v) {
  const int n = ok(e, liz_rx_chunks(h<liz_rx>(v), nullptr, 0));
  if (n < 0) return nullptr;
  jbyteArray a = e->NewByteArray(n);
  if (n) { Crit c(e, a); liz_rx_chunks(h<liz_rx>(v), c.bytes(), n); }
  return a;
}
jstring rxMeta(JNIEnv* e, jclass, jlong v, jint which) { return e->NewStringUTF(liz_rx_meta(h<liz_rx>(v), which)); }
jbyteArray rxData(JNIEnv* e, jclass, jlong v) {
  const uint8_t* p;
  size_t n;
  if (liz_rx_data(h<liz_rx>(v), &p, &n) < 0) return nullptr;
  jbyteArray a = e->NewByteArray(jsize(n));
  e->SetByteArrayRegion(a, 0, jsize(n), reinterpret_cast<const jbyte*>(p));
  return a;
}
void rxClear(JNIEnv*, jclass, jlong v) { liz_rx_clear(h<liz_rx>(v)); }

// ---- layer 4, the engines ----------------------------------------------------------------------------------------------

// ai: a receiver and the Kotlin callbacks it calls: release(tag) (dev.lizard.Receiver.Release) and log(line)
struct JReceiver {
  liz_receiver* r = nullptr;
  jobject release = nullptr, log = nullptr;
  jmethodID released = nullptr, logged = nullptr;
  std::string stats, file;
};
void onRelease(void* user, uint64_t tag) {
  auto* j = static_cast<JReceiver*>(user);
  if (JNIEnv* e = envHere()) { e->CallVoidMethod(j->release, j->released, jlong(tag)); if (e->ExceptionCheck()) e->ExceptionClear(); }
}
void onLog(void* user, const char* line) {
  auto* j = static_cast<JReceiver*>(user);
  if (!j->log) return;
  if (JNIEnv* e = envHere()) {
    jstring s = e->NewStringUTF(line);
    e->CallVoidMethod(j->log, j->logged, s);
    e->DeleteLocalRef(s);
    if (e->ExceptionCheck()) e->ExceptionClear();
  }
}
jlong receiverNew(JNIEnv* e, jclass, jstring assets, jstring cacheDir, jstring storeDir, jstring device, jstring precision,
                  jint decoder, jint cpuThreads, jint layout, jobject release, jobject log) {
  Str a(e, assets), c(e, cacheDir), st(e, storeDir), d(e, device), p(e, precision);
  auto* j = new JReceiver();
  j->release = e->NewGlobalRef(release);
  j->released = e->GetMethodID(e->GetObjectClass(release), "released", "(J)V");
  if (log) { j->log = e->NewGlobalRef(log); j->logged = e->GetMethodID(e->GetObjectClass(log), "log", "(Ljava/lang/String;)V"); }
  liz_receiver_config cfg{};
  cfg.assets = a.c; cfg.cache_dir = c.c; cfg.store_dir = st.c; cfg.device = d.c; cfg.precision = p.c;
  cfg.decoder = decoder; cfg.cpu_threads = cpuThreads; cfg.layout = layout; cfg.log = log ? onLog : nullptr; cfg.user = j;
  // ai: the C config has no user for release beyond cfg.user: both callbacks take the same JReceiver
  j->r = liz_receiver_new(&cfg, onRelease);
  if (!j->r) {
    const int code = liz_last_error_code();
    e->DeleteGlobalRef(j->release);
    if (j->log) e->DeleteGlobalRef(j->log);
    delete j;
    throwFor(e, code < 0 ? code : LIZ_E_INTERNAL);
    return 0;
  }
  return reinterpret_cast<jlong>(j);
}
void receiverFree(JNIEnv* e, jclass, jlong v) {
  auto* j = h<JReceiver>(v);
  if (!j) return;
  liz_receiver_free(j->r);   // ai: its threads joined, every frame released, before the callbacks' references go
  e->DeleteGlobalRef(j->release);
  if (j->log) e->DeleteGlobalRef(j->log);
  delete j;
}
jint receiverPush(JNIEnv* e, jclass, jlong v, jbyteArray luma, jint offset, jint w, jint hh, jint stride, jlong ts, jlong tag) {
  int rc;
  if (!holds(e, luma, spanOf(offset, w, hh, stride, 1), "luma")) return 0;
  // ai: not a critical region: the receiver calls back into Java inside push (release, log), which a critical region
  // ai: forbids (CheckJNI aborted a debuggable app at the first push, a JVM could deadlock on its GC)
  { Elems l(e, luma); if (!l.p) return 0; rc = liz_receiver_push(h<JReceiver>(v)->r, l.bytes() + offset, w, hh, stride, ts, uint64_t(tag)); }
  return ok(e, rc);
}
jint receiverPushBuffer(JNIEnv* e, jclass, jlong v, jobject luma, jint offset, jint w, jint hh, jint stride, jlong ts, jlong tag) {
  if (!holdsBuf(e, luma, spanOf(offset, w, hh, stride, 1), "luma")) return 0;
  return ok(e, liz_receiver_push(h<JReceiver>(v)->r, direct(e, luma) + offset, w, hh, stride, ts, uint64_t(tag)));
}
jint receiverPushHardwareBuffer(JNIEnv* e, jclass, jlong v, jobject hb, jint w, jint hh, jlong ts, jlong tag) {
#ifdef __ANDROID__
  AHardwareBuffer* b = AHardwareBuffer_fromHardwareBuffer(e, hb);
  return ok(e, liz_receiver_push_hardware_buffer(h<JReceiver>(v)->r, b, w, hh, ts, uint64_t(tag)));
#else
  (void)hb;
  return ok(e, liz_receiver_push_hardware_buffer(h<JReceiver>(v)->r, nullptr, w, hh, ts, uint64_t(tag)));
#endif
}
jboolean receiverWantsLuma(JNIEnv* e, jclass, jlong v) { return ok(e, liz_receiver_wants_luma(h<JReceiver>(v)->r)) > 0; }
jstring receiverStats(JNIEnv* e, jclass, jlong v) { return e->NewStringUTF(liz_receiver_stats(h<JReceiver>(v)->r)); }
jdoubleArray receiverSeries(JNIEnv* e, jclass, jlong v, jdouble since) {
  std::vector<double> buf(1 + 8 * 200);
  const int n = ok(e, liz_receiver_series(h<JReceiver>(v)->r, since, buf.data(), int(buf.size())));
  if (n < 0) return nullptr;
  jdoubleArray a = e->NewDoubleArray(n);
  e->SetDoubleArrayRegion(a, 0, n, buf.data());
  return a;
}
void receiverSoon(JNIEnv*, jclass, jlong v, jboolean on) { liz_receiver_soon(h<JReceiver>(v)->r, on ? 1 : 0); }
void receiverBatchCap(JNIEnv*, jclass, jlong v, jint n) { liz_receiver_batch_cap(h<JReceiver>(v)->r, n); }
jstring receiverFile(JNIEnv* e, jclass, jlong v) { return e->NewStringUTF(liz_receiver_file(h<JReceiver>(v)->r)); }
void receiverClear(JNIEnv*, jclass, jlong v) { liz_receiver_clear(h<JReceiver>(v)->r); }
void receiverCameraClosed(JNIEnv*, jclass, jlong v) { liz_receiver_camera_closed(h<JReceiver>(v)->r); }

// ai: a sender and the file's bytes it reads (copied in)
jlong senderNew(JNIEnv* e, jclass, jbyteArray bytes, jstring name, jstring type) {
  Str n(e, name), t(e, type);
  liz_sender* s;
  if (!bytes) s = liz_sender_new(nullptr, 0, n.c, t.c, 0);
  else {
    const jsize len = e->GetArrayLength(bytes);
    Elems b(e, bytes);   // ai: hashes the whole file
    if (len && !b.p) return 0;
    s = liz_sender_new(b.bytes(), size_t(len), n.c, t.c, LIZ_TX_COPY);
  }
  return made(e, s);
}
void senderFree(JNIEnv*, jclass, jlong v) { liz_sender_free(h<liz_sender>(v)); }
void senderConfigure(JNIEnv* e, jclass, jlong v, jint blocks, jint ring, jint fps, jint codes, jint painter, jint threads, jstring assets, jstring device) {
  Str a(e, assets), d(e, device);
  liz_format f{blocks, ring, fps, codes};
  ok(e, liz_sender_configure(h<liz_sender>(v), &f, painter, threads, a.c, d.c));
}
jboolean senderReady(JNIEnv* e, jclass, jlong v) { return ok(e, liz_sender_ready(h<liz_sender>(v))) > 0; }
jboolean senderTake(JNIEnv* e, jclass, jlong v, jbyteArray out, jint stride, jint fmt) {
  liz_geometry g{};
  if (ok(e, liz_sender_geometry(h<liz_sender>(v), &g)) < 0) return false;
  if (!holds(e, out, spanOf(0, g.width, g.height, stride, bppOf(fmt)), "out")) return false;
  int rc;
  { Crit o(e, out); rc = liz_sender_take(h<liz_sender>(v), o.bytes(), stride, liz_pixfmt(fmt)); }
  return ok(e, rc) > 0;
}
jboolean senderTakeBuffer(JNIEnv* e, jclass, jlong v, jobject out, jint stride, jint fmt) {
  liz_geometry g{};
  if (ok(e, liz_sender_geometry(h<liz_sender>(v), &g)) < 0) return false;
  if (!holdsBuf(e, out, spanOf(0, g.width, g.height, stride, bppOf(fmt)), "out")) return false;
  return ok(e, liz_sender_take(h<liz_sender>(v), direct(e, out), stride, liz_pixfmt(fmt))) > 0;
}
jstring senderStats(JNIEnv* e, jclass, jlong v) { return e->NewStringUTF(liz_sender_stats(h<liz_sender>(v))); }

#define M(name, sig) {const_cast<char*>(#name), const_cast<char*>(sig), reinterpret_cast<void*>(name)}
const JNINativeMethod METHODS[] = {
    M(version, "()Ljava/lang/String;"),
    M(abi, "()I"),
    M(simd, "()Z"),
    M(ringCells, "(I)I"),
    M(geometry, "(IIII[I)V"),
    M(roomFor, "(II)D"),
    M(pick, "(DDIII)I"),
    M(streamFill, "(I[B)V"),
    M(layoutRects, "(III[I)I"),
    M(encoderNew, "(IIII)J"),
    M(encoderFree, "(J)V"),
    M(encoderPaint, "(J[BI[BII)V"),
    M(encoderPaintBuffer, "(J[BILjava/nio/ByteBuffer;II)V"),
    M(decoderNew, "(I)J"),
    M(decoderFree, "(J)V"),
    M(decoderMaxBlocks, "(J)I"),
    M(decode, "(J[BIIIII[I[B[I[F)I"),
    M(decodeBuffer, "(JLjava/nio/ByteBuffer;IIIII[I[B[I[F)I"),
    M(txNew, "([BLjava/lang/String;Ljava/lang/String;)J"),
    M(txNewPath, "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)J"),
    M(txNewTest, "(I)J"),
    M(txFree, "(J)V"),
    M(txNext, "(JI[B)I"),
    M(txInfo, "(J[J[B)V"),
    M(rxNew, "(Ljava/lang/String;J)J"),
    M(rxFree, "(J)V"),
    M(rxFrame, "(J[BI[I)V"),
    M(rxProgress, "(J[I[J[D)V"),
    M(rxChunks, "(J)[B"),
    M(rxMeta, "(JI)Ljava/lang/String;"),
    M(rxData, "(J)[B"),
    M(rxClear, "(J)V"),
    M(receiverNew, "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IIILdev/lizard/Receiver$Release;Ldev/lizard/Receiver$Log;)J"),
    M(receiverFree, "(J)V"),
    M(receiverPush, "(J[BIIIIJJ)I"),
    M(receiverPushBuffer, "(JLjava/nio/ByteBuffer;IIIIJJ)I"),
    M(receiverPushHardwareBuffer, "(JLjava/lang/Object;IIJJ)I"),
    M(receiverWantsLuma, "(J)Z"),
    M(receiverStats, "(J)Ljava/lang/String;"),
    M(receiverSeries, "(JD)[D"),
    M(receiverSoon, "(JZ)V"),
    M(receiverBatchCap, "(JI)V"),
    M(receiverFile, "(J)Ljava/lang/String;"),
    M(receiverClear, "(J)V"),
    M(receiverCameraClosed, "(J)V"),
    M(senderNew, "([BLjava/lang/String;Ljava/lang/String;)J"),
    M(senderFree, "(J)V"),
    M(senderConfigure, "(JIIIIIILjava/lang/String;Ljava/lang/String;)V"),
    M(senderReady, "(J)Z"),
    M(senderTake, "(J[BII)Z"),
    M(senderTakeBuffer, "(JLjava/nio/ByteBuffer;II)Z"),
    M(senderStats, "(J)Ljava/lang/String;"),
};
#undef M

}  // namespace

extern "C" LIZ_API jint JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* e = nullptr;
  gVm = vm;
  if (vm->GetEnv(reinterpret_cast<void**>(&e), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
  jclass native = e->FindClass("dev/lizard/Native");
  jclass ex = e->FindClass("dev/lizard/LizardException");
  if (!native || !ex) return JNI_ERR;
  gLizardException = static_cast<jclass>(e->NewGlobalRef(ex));
  gLizardExceptionInit = e->GetMethodID(gLizardException, "<init>", "(ILjava/lang/String;)V");
  if (!gLizardExceptionInit) return JNI_ERR;
  if (e->RegisterNatives(native, METHODS, sizeof METHODS / sizeof *METHODS) != JNI_OK) return JNI_ERR;
  return JNI_VERSION_1_6;
}
