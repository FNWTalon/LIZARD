// ai: What the desktop library's JNI shares (2026-10-25): the strings as the text they are (liblizard/api/jni_text.h),
// ai: and the error channel: the last refusal of a call that returns a handle or null (txCreate, presentStart,
// ai: portalPick) is one process-wide string, Native.txError()'s; the last writer wins across threads. The receive
// ai: session's JNI is liblizard's (api/jni_session.cpp, dev.lizard.SessionNative), registered at the library's load.
#pragma once
#include <jni.h>

#include <string>

#include "jni_text.h"

// ai: a function of dev.lizard.desktop.Native: FN(jstring, txStats)(JNIEnv* e, jclass, jlong h) { ... }
#define FN(ret, name) extern "C" JNIEXPORT ret JNICALL Java_dev_lizard_desktop_Native_##name

namespace lizard::jni {

void setError(const std::string& m);
std::string lastError();

}  // namespace lizard::jni
