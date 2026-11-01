// ai: The receive session's JNI (jni_session.cpp, 2026-10-31): dev.lizard.SessionNative's methods registered on the
// ai: class (RegisterNatives), from the JNI_OnLoad of whichever library carries them: the binding's liblizard
// ai: (api/jni.cpp) and the desktop app's library (lizard-desktop/native/jni_util.cpp). False with a Java exception
// ai: pending where the class or a method is missing.
#pragma once
#include <jni.h>

namespace lizard::jni {

bool registerSession(JNIEnv* e);

}  // namespace lizard::jni
