// ai: What the desktop library's two JNI faces share (2026-10-25): tx_jni.cpp's (the sender, the presenter, the file
// ai: chooser) and rx_jni.cpp's (the receiver), both dev.lizard.desktop.Native's.
// ai: Strings cross as the text they are. JNI's own UTF calls (GetStringUTFChars, NewStringUTF) speak modified UTF-8,
// ai: in which a character past U+FFFF (an emoji in a file's name) is two 3-byte surrogates and a 4-byte sequence is
// ai: not valid input, so a path from the chooser reached the file system mangled and a name read off the light came
// ai: back wrong or was refused. Here a jstring's UTF-16 becomes UTF-8 and back by hand: a surrogate pair is one
// ai: 4-byte sequence, a lone surrogate or bytes that are not UTF-8 become U+FFFD (a maximal invalid subpart one each).
// ai: The error channel: the last refusal of a call that returns a handle or null (txCreate, presentStart,
// ai: portalPick, rxCreate) is one process-wide string, Native.txError()'s; the last writer wins across threads.
#pragma once
#include <jni.h>

#include <cstddef>
#include <string>

// ai: a function of dev.lizard.desktop.Native: FN(jstring, txStats)(JNIEnv* e, jclass, jlong h) { ... }
#define FN(ret, name) extern "C" JNIEXPORT ret JNICALL Java_dev_lizard_desktop_Native_##name

namespace lizard::jni {

// ai: UTF-16 to UTF-8, the conversion str makes (tx_jni.cpp's too, for a wide environment value on Windows); jstr's
// ai: way back is jni_util.cpp's own
std::string utf8(const jchar* s, size_t n);

// ai: a jstring's text as UTF-8 ("" for null)
std::string str(JNIEnv* e, jstring s);
// ai: UTF-8 as a jstring (null only with the VM's OutOfMemoryError pending)
jstring jstr(JNIEnv* e, const std::string& s);

void setError(const std::string& m);
std::string lastError();

}  // namespace lizard::jni
