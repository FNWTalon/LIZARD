// ai: jni_util.h: the shared error channel.
#include "jni_util.h"

#include <mutex>

#include "jni_session.h"

namespace lizard::jni {

namespace {
std::mutex gErrorMu;
std::string gError;
}  // namespace

void setError(const std::string& m) {
  std::lock_guard<std::mutex> l(gErrorMu);
  gError = m;
}

std::string lastError() {
  std::lock_guard<std::mutex> l(gErrorMu);
  return gError;
}

}  // namespace lizard::jni

// ai: The library's load: the receive session's methods (dev.lizard.SessionNative, liblizard/api/jni_session.cpp)
// ai: registered, the sender's found by their Java_ names as before.
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* e = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&e), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
  if (!lizard::jni::registerSession(e)) return JNI_ERR;
  return JNI_VERSION_1_6;
}
